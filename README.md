# FastAPI Docs Agent

## Corpus Freeze
- **Source:** https://github.com/tiangolo/fastapi
- **Version:** 0.136.3
- **Commit SHA:** 82064857539e6286522c347b4b11331b48dd2378
- **Frozen:** 2025-06-12

Any eval numbers in this repo are claims against this exact corpus state.

## Corpus Integrity
- **Corpus hash:** 69e3a0cbd84eb6276470ceaeda63b5e2fbf590c7e6b1cf9c87bb40efade4f930
- **Files:** 153 markdown files, 433 code includes resolved
- **Built with:** `python scripts/build_corpus.py`

To verify the corpus hasn't changed:
    find corpus -type f | sort | xargs sha256sum | sha256sum
