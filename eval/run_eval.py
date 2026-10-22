"""Compare retrieval modes on hand-written questions with known relevant indices.

Metrics, averaged over questions:
  hit@k  share of questions with at least one relevant index in the top k
  MRR    mean of 1/rank of the first relevant index (0 if none in the top 10)

    uv run python eval/run_eval.py
    uv run python eval/run_eval.py --show-misses hybrid
"""

import argparse
import json
from pathlib import Path

from earth_copilot.rag.retrieve import search

HERE = Path(__file__).parent
MODES = ["dense", "sparse", "hybrid"]
DEPTH = 10


def first_relevant_rank(found: list[str], relevant: set[str]) -> int | None:
    return next((rank for rank, name in enumerate(found, 1) if name in relevant), None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-misses", choices=MODES, help="list questions missed at top 3")
    args = parser.parse_args()

    questions = [json.loads(line) for line in (HERE / "rag_questions.jsonl").open(encoding="utf-8")]
    results = {}
    for mode in MODES:
        ranks, misses = [], []
        for q in questions:
            found = [hit.short_name for hit in search(q["question"], k=DEPTH, mode=mode)]
            rank = first_relevant_rank(found, set(q["relevant"]))
            ranks.append(rank)
            if rank is None or rank > 3:
                misses.append((q["question"], found[:3]))
        n = len(ranks)
        results[mode] = {
            "hit@1": sum(r is not None and r <= 1 for r in ranks) / n,
            "hit@3": sum(r is not None and r <= 3 for r in ranks) / n,
            "hit@5": sum(r is not None and r <= 5 for r in ranks) / n,
            "MRR": sum(1 / r for r in ranks if r) / n,
        }
        if mode == args.show_misses:
            for question, top in misses:
                print(f"MISS  {question}\n      got {top}")

    print(f"\n{len(questions)} questions\n")
    print("| mode | hit@1 | hit@3 | hit@5 | MRR |")
    print("|---|---|---|---|---|")
    for mode, m in results.items():
        print(f"| {mode} | {m['hit@1']:.2f} | {m['hit@3']:.2f} | {m['hit@5']:.2f} | {m['MRR']:.2f} |")

    out = HERE / "results" / "rag_retrieval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
