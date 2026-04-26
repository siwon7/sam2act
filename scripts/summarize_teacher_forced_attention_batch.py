#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_summaries(input_dir: Path, stem_prefix: str, episodes: int) -> list[dict]:
    payloads = []
    for ep in range(episodes):
        path = input_dir / f"{stem_prefix}_ep{ep}.json"
        if not path.exists():
            continue
        with path.open() as f:
            payloads.append(json.load(f))
    return payloads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--stem-prefix", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    payloads = load_summaries(Path(args.input_dir), args.stem_prefix, args.episodes)
    step_rank_counts: dict[int, dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))
    step_rank_scores: dict[int, dict[int, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for payload in payloads:
        for item in payload["summary"]:
            step = int(item["step"])
            for rank, past in enumerate(item["top_past_steps"][: args.topk], start=1):
                ps = int(past["past_step"])
                sc = float(past["score_mean"])
                step_rank_counts[step][rank][ps] += 1
                step_rank_scores[step][rank][ps].append(sc)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "rank",
                "past_step",
                "count",
                "episodes_seen",
                "frequency",
                "mean_score",
            ]
        )
        for step in sorted(step_rank_counts):
            for rank in range(1, args.topk + 1):
                total = sum(step_rank_counts[step][rank].values())
                for past_step, count in step_rank_counts[step][rank].most_common():
                    scores = step_rank_scores[step][rank][past_step]
                    writer.writerow(
                        [
                            step,
                            rank,
                            past_step,
                            count,
                            total,
                            count / total if total else 0.0,
                            sum(scores) / len(scores) if scores else 0.0,
                        ]
                    )

    output_md = Path(args.output_md)
    lines = [
        f"# Teacher-Forced Attention Batch Summary",
        "",
        f"- episodes: {len(payloads)}",
        f"- topk: {args.topk}",
        "",
    ]
    for step in sorted(step_rank_counts):
        lines.append(f"## Step {step}")
        for rank in range(1, args.topk + 1):
            total = sum(step_rank_counts[step][rank].values())
            items = []
            for past_step, count in step_rank_counts[step][rank].most_common(5):
                scores = step_rank_scores[step][rank][past_step]
                mean_score = sum(scores) / len(scores) if scores else 0.0
                items.append(
                    f"{past_step} ({count}/{total}, {mean_score:.4f})"
                )
            lines.append(f"- top{rank}: " + (", ".join(items) if items else "n/a"))
        lines.append("")
    output_md.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
