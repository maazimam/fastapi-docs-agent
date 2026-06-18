"""
chunker.py

Splits every markdown file in corpus/prose/ into sections
by splitting on ## and ### headers.

Each chunk is a dict:
    {
        "id":      "prose/tutorial/path-params.md#path-parameters",
        "source":  "prose/tutorial/path-params.md",
        "heading": "Path Parameters",
        "text":    "## Path Parameters\n\nYou can declare..."
    }
"""

import re
from pathlib import Path

CORPUS_DIR = Path("corpus/prose")
MIN_CHARS = 100  # skip sections too short to be meaningful

SKIP_FILES = {"_llm-test.md"}

ANCHOR_TAG_RE = re.compile(r"\{.*?\}")


def chunk_file(md_path: Path) -> list[dict]:
    """Split one markdown file into sections by ## and ### headers."""
    text = md_path.read_text(encoding="utf-8")
    rel_path = str(md_path.relative_to(Path("corpus")))

    # split on lines that start with ## or ###
    parts = re.split(r"\n(?=#{2,3} )", text)

    chunks = []
    for part in parts:
        part = part.strip()
        part = ANCHOR_TAG_RE.sub("", part)
        if len(part) < MIN_CHARS:
            continue

        first_line = part.split("\n")[0]
        heading = first_line.lstrip("#").strip()
        heading = ANCHOR_TAG_RE.sub("", heading).strip()  # remove { #anchor } tags
        slug = heading.lower().replace(" ", "-")

        chunks.append({
            "id": f"{rel_path}#{slug}",
            "source": rel_path,
            "heading": heading,
            "text": part,
        })

    return chunks


def load_all_chunks() -> list[dict]:
    all_chunks = []
    for md_file in sorted(CORPUS_DIR.rglob("*.md")):
        if md_file.name in SKIP_FILES:
            continue
        all_chunks.extend(chunk_file(md_file))
    return all_chunks


if __name__ == "__main__":
    chunks = load_all_chunks()
    print(f"Total chunks: {len(chunks)}")
    print("\nSample chunk:")
    print(f"  id:      {chunks[10]['id']}")
    print(f"  source:  {chunks[10]['source']}")
    print(f"  heading: {chunks[10]['heading']}")
    print(f"  text:    {chunks[10]['text'][:200]}...")
