"""Ask the catalogue a question.

    uv run python scripts/ask_docs.py "Which index should I use to map flooded areas?"
    uv run python scripts/ask_docs.py "burned forest" --no-llm --mode sparse
"""

import argparse

from earth_copilot.rag.answer import answer
from earth_copilot.rag.retrieve import search


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--mode", choices=["dense", "sparse", "hybrid"], default="hybrid")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--no-llm", action="store_true", help="only show what retrieval finds")
    args = parser.parse_args()

    if args.no_llm:
        for i, hit in enumerate(search(args.question, k=args.k, mode=args.mode), 1):
            print(f"[{i}] {hit.short_name:10} {hit.score:.3f}  {hit.long_name} ({hit.domain})")
        return

    result = answer(args.question, k=args.k, mode=args.mode)
    print(result.text)
    print("\nSources:")
    for i, hit in enumerate(result.sources, 1):
        print(f"[{i}] {hit.short_name} — {hit.long_name} — {hit.reference}")


if __name__ == "__main__":
    main()
