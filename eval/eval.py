from pathlib import Path
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CORPUS_ROOT = Path(__file__).resolve().parent.parent / "corpus"


def citation_exists(citation: str) -> bool:
    return (CORPUS_ROOT / citation).exists()


def check_citations(citations: list[str]) -> dict:
    valid = [c for c in citations if citation_exists(c)]
    invalid = [c for c in citations if not citation_exists(c)]

    return {
        "valid_count": len(valid),
        "invalid_count": len(invalid),
        "invalid_citations": invalid,
        "has_hallucinations": len(invalid) > 0,
    }


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

JUDGE_MODEL = "gpt-4o-mini"
INPUT_COST   = 0.00000015   # $ per input token
OUTPUT_COST  = 0.00000060   # $ per output token

JUDGE_PROMPT = """You are a strict factual judge evaluating whether an AI answer is faithful to a reference answer.

Question: {question}

Reference answer (ground truth):
{expected_answer}

System answer:
{system_answer}

Score the system answer on faithfulness to the reference on a scale of 0 to 2:
  2 = fully faithful, all claims consistent with the reference
  1 = partially faithful, some claims consistent but some missing or slightly off
  0 = unfaithful, contradicts or ignores the reference

Reply with ONLY a single integer: 0, 1, or 2. No explanation."""

def judge_faithfulness(question: str, expected: str, system_answer: str) -> tuple[int, float]:
    """
    Ask the LLM judge to score faithfulness.
    Returns (score 0-2, cost in USD).
    """
    prompt = JUDGE_PROMPT.format(
        question=question,
        expected_answer=expected,
        system_answer=system_answer,
    )
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1,
        temperature=0,
    )
    raw   = response.choices[0].message.content.strip()
    score = int(raw) if raw in {"0", "1", "2"} else 0
    cost  = (response.usage.prompt_tokens     * INPUT_COST +
             response.usage.completion_tokens * OUTPUT_COST)
    return score, cost

import json
import time

GROUND_TRUTH = Path("eval/ground_truth.json")


def run_eval(answer_fn) -> dict:
    """
    Run answer_fn against every question in ground_truth.json.
    Returns per-question results and aggregate metrics.
    """
    ground_truth = json.loads(GROUND_TRUTH.read_text())

    results      = []
    judge_costs  = []
    latencies_ms = []

    for item in ground_truth:
        qid            = item["id"]
        question       = item["question"]
        category       = item["category"]
        expected       = item.get("expected_answer")
        should_abstain = item["should_abstain"]

        # call the answer function and time it
        t0 = time.perf_counter()
        try:
            response = answer_fn(question)
        except Exception as e:
            response = {"answer": None, "citations": [], "error": str(e)}
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(latency_ms)

        system_answer = response.get("answer")
        citations     = response.get("citations", [])
        did_abstain   = system_answer is None

        # deterministic: was the abstention decision correct?
        abstention_correct = (did_abstain == should_abstain)

        # deterministic: citation check
        cite_result = check_citations(citations)

        # LLM judge: only for answerable questions that got an answer
        judge_score = None
        judge_cost  = 0.0
        if not should_abstain and not did_abstain and expected:
            judge_score, judge_cost = judge_faithfulness(
                question, expected, system_answer
            )
            judge_costs.append(judge_cost)

        results.append({
            "id":                 qid,
            "category":           category,
            "should_abstain":     should_abstain,
            "did_abstain":        did_abstain,
            "abstention_correct": abstention_correct,
            "citations":          citations,
            "citation_valid":     cite_result["valid_count"],
            "citation_invalid":   cite_result["invalid_count"],
            "has_hallucination":  cite_result["has_hallucination"],
            "judge_score":        judge_score,
            "latency_ms":         latency_ms,
        })

        status = "✓" if abstention_correct else "✗"
        print(f"[{status}] {qid} ({category}) | abstain={did_abstain} | "
              f"citations={len(citations)} | judge={judge_score} | "
              f"{latency_ms:.0f}ms")

    metrics = compute_metrics(results, judge_costs, latencies_ms)
    return {"results": results, "metrics": metrics}

import statistics


def compute_metrics(results, judge_costs, latencies_ms) -> dict:

    # ── abstention precision / recall — ALL categories ────────────────────
    # tp = correctly abstained, fp = wrongly abstained, fn = should have abstained but didn't
    tp = sum(1 for r in results if r["did_abstain"] and r["should_abstain"])
    fp = sum(1 for r in results if r["did_abstain"] and not r["should_abstain"])
    fn = sum(1 for r in results if not r["did_abstain"] and r["should_abstain"])

    abstention_precision = tp / (tp + fp) if (tp + fp) > 0 else None
    abstention_recall    = tp / (tp + fn) if (tp + fn) > 0 else None

    # ── abstention precision / recall — IN-DOMAIN-UNCOVERED only ─────────
    idc    = [r for r in results if r["category"] == "in_domain_uncovered"]
    tp_idc = sum(1 for r in idc if r["did_abstain"] and r["should_abstain"])
    fp_idc = sum(1 for r in idc if r["did_abstain"] and not r["should_abstain"])
    fn_idc = sum(1 for r in idc if not r["did_abstain"] and r["should_abstain"])

    abstention_precision_idc = tp_idc / (tp_idc + fp_idc) if (tp_idc + fp_idc) > 0 else None
    abstention_recall_idc    = tp_idc / (tp_idc + fn_idc) if (tp_idc + fn_idc) > 0 else None

    # ── citation accuracy ─────────────────────────────────────────────────
    total_citations = sum(r["citation_valid"] + r["citation_invalid"] for r in results)
    valid_citations = sum(r["citation_valid"] for r in results)
    citation_accuracy = valid_citations / total_citations if total_citations > 0 else None

    # ── hallucinated citation rate ────────────────────────────────────────
    answered = [r for r in results if not r["did_abstain"]]
    hallucination_rate = (
        sum(1 for r in answered if r["has_hallucination"]) / len(answered)
        if answered else None
    )

    # ── faithfulness (LLM judge, normalised 0-1) ──────────────────────────
    judged = [r for r in results if r["judge_score"] is not None]
    faithfulness = (
        statistics.mean(r["judge_score"] / 2.0 for r in judged)
        if judged else None
    )

    # ── cost ──────────────────────────────────────────────────────────────
    cost_per_query = sum(judge_costs) / len(results) if results else None

    # ── latency ───────────────────────────────────────────────────────────
    latency_p50 = statistics.median(latencies_ms) if latencies_ms else None
    latency_p95 = (
        sorted(latencies_ms)[int(len(latencies_ms) * 0.95)]
        if latencies_ms else None
    )

    return {
        "abstention_precision":       abstention_precision,
        "abstention_recall":          abstention_recall,
        "abstention_precision_idc":   abstention_precision_idc,
        "abstention_recall_idc":      abstention_recall_idc,
        "citation_accuracy":          citation_accuracy,
        "hallucinated_citation_rate": hallucination_rate,
        "faithfulness":               faithfulness,
        "cost_per_query_usd":         cost_per_query,
        "latency_p50_ms":             latency_p50,
        "latency_p95_ms":             latency_p95,
        "n_questions":                len(results),
        "n_judged":                   len(judged),
    }

def print_metrics(metrics: dict):
    print("\n" + "=" * 55)
    print(f"{'METRIC':<35} {'VALUE':>18}")
    print("=" * 55)
    for k, v in metrics.items():
        if v is None:
            print(f"  {k:<33} {'N/A':>18}")
        elif isinstance(v, float):
            print(f"  {k:<33} {v:>17.4f}")
        else:
            print(f"  {k:<33} {v:>18}")
    print("=" * 55)

import argparse
import importlib


def load_answer_fn(dotted_path: str):
    """
    Import an answer function from a dotted path.
    e.g. 'baseline.answer' imports answer() from baseline/answer.py
    """
    module_path, fn_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, fn_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastAPI docs agent eval harness.")
    parser.add_argument(
        "--answer-fn",
        required=True,
        help="Dotted path to answer function e.g. baseline.answer",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write JSON results e.g. results/baseline.json",
    )
    args = parser.parse_args()

    fn  = load_answer_fn(args.answer_fn)
    out = run_eval(fn)

    print_metrics(out["metrics"])

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"\nResults written to {args.output}")