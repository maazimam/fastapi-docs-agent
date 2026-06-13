"""
build_corpus.py

Reads the frozen FastAPI docs at scripts/fastapi-repo/docs/en/docs,
resolves all {* docs_src/... *} code includes by inlining the actual
Python files, and writes clean .md files to corpus/prose/.

Run from the project root:
    python scripts/build_corpus.py
"""

import re
import shutil
from pathlib import Path

REPO_ROOT = Path("scripts/fastapi-repo")
DOCS_DIR = REPO_ROOT / "docs/en/docs"
SRC_DIR = REPO_ROOT / "docs_src"
OUT_DIR = Path("corpus/prose")

# e.g. {* ../../docs_src/foo/bar.py *}
#      {* ../../docs_src/foo/bar.py hl[1,3:5] *}
#      {* ../../docs_src/foo/bar.py hl[1] title["app/main.py"] *}
INCLUDE_RE = re.compile(
    r"\{\*\s*((?:\.\./)*(?:docs_src|fastapi)/\S+?)"
    r"(?:\s+ln\[[^\]]*\])?"
    r"(?:\s+hl\[[^\]]*\])?"
    r'(?:\s+title\[[^\]]*\])?'
    r"\s*\*\}"
)

SKIP_DIRS = {"css", "js", "img"}

LANG_BY_EXT = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".html": "html",
    ".css": "css",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".sh": "bash",
}


def resolve_include_file(include_path: str) -> Path | None:
    """Resolve '../../docs_src/foo.py' or '../../fastapi/openapi/docs.py'."""
    for marker, base in (("docs_src/", SRC_DIR), ("fastapi/", REPO_ROOT / "fastapi")):
        idx = include_path.find(marker)
        if idx != -1:
            return base / include_path[idx + len(marker) :]
    return None


def resolve_includes(md_text: str) -> tuple[str, int, int]:
    resolved = 0
    missing = 0

    def replacer(match: re.Match[str]) -> str:
        nonlocal resolved, missing
        src_file = resolve_include_file(match.group(1))
        if src_file is None or not src_file.exists():
            missing += 1
            return f"<!-- include not found: {match.group(1)} -->"

        code = src_file.read_text(encoding="utf-8")
        lang = LANG_BY_EXT.get(src_file.suffix, src_file.suffix.lstrip(".") or "text")
        resolved += 1
        return f"```{lang}\n{code}\n```"

    return INCLUDE_RE.sub(replacer, md_text), resolved, missing


def build() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    files_written = 0
    includes_resolved = 0
    includes_missing = 0

    for md_file in sorted(DOCS_DIR.rglob("*.md")):
        parts = md_file.relative_to(DOCS_DIR).parts
        if any(part in SKIP_DIRS for part in parts):
            continue

        text = md_file.read_text(encoding="utf-8")
        resolved_text, found, missing = resolve_includes(text)
        includes_resolved += found
        includes_missing += missing

        rel = md_file.relative_to(DOCS_DIR)
        out_file = OUT_DIR / rel
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(resolved_text, encoding="utf-8")
        files_written += 1

    print(f"Done. {files_written} files written to {OUT_DIR}/")
    print(f"      {includes_resolved} code includes resolved.")
    if includes_missing:
        print(f"      {includes_missing} includes missing (left as HTML comments).")


if __name__ == "__main__":
    build()
