"""Step 2: download the spectral-index catalogue and index it in Qdrant.

    ollama pull nomic-embed-text
    uv run python scripts/build_index.py
"""

import time

from earth_copilot.rag.ingest import ingest


def main() -> None:
    started = time.perf_counter()
    count = ingest()
    print(f"Indexed {count} spectral indices in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
