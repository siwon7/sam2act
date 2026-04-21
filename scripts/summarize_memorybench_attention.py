#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PUT_BLOCK_BACK_LABELS = {
    0: "move_to_block",
    1: "lift_block",
    2: "move_to_center",
    3: "place_center",
    4: "rise_for_button",
    5: "move_over_button",
    6: "press_button",
    7: "rise_over_center",
    8: "grasp_block",
    9: "lift_block_again",
    10: "move_to_initial_slot",
    11: "place_at_initial_slot",
}

PUT_BLOCK_BACK_EXPECTED = {
    7: [2, 3, 6],
    8: [3, 2, 6],
    9: [8, 7, 3],
    10: [1, 0, 6],
    11: [0, 1, 10],
}


def get_labels(task: str) -> dict[int, str]:
    if task == "put_block_back":
        return PUT_BLOCK_BACK_LABELS
    return {}


def get_expected(task: str) -> dict[int, list[int]]:
    if task == "put_block_back":
        return PUT_BLOCK_BACK_EXPECTED
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    payload = json.loads(Path(args.input_json).read_text())
    task = payload.get("task", "")
    labels = get_labels(task)
    expected = get_expected(task)
    summary = payload["summary"]

    rows = []
    for item in summary:
        step = int(item["step"])
        top = item["top_past_steps"][: args.topk]
        flat = {
            "step": step,
            "step_label": labels.get(step, ""),
            "expected_refs": ",".join(str(x) for x in expected.get(step, [])),
        }
        for rank, past in enumerate(top, start=1):
            flat[f"top{rank}_past_step"] = int(past["past_step"])
            flat[f"top{rank}_past_label"] = labels.get(int(past["past_step"]), "")
            flat[f"top{rank}_score"] = float(past["score_mean"])
        rows.append(flat)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["step", "step_label", "expected_refs"]
    for rank in range(1, args.topk + 1):
        fieldnames.extend([f"top{rank}_past_step", f"top{rank}_past_label", f"top{rank}_score"])

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# MemoryBench attention top{args.topk} summary",
        "",
        f"- task: `{task}`",
        f"- source: `{args.input_json}`",
        "",
        "| step | label | expected | top1 | top2 | top3 |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        tops = []
        for rank in range(1, args.topk + 1):
            past = row.get(f"top{rank}_past_step", "")
            label = row.get(f"top{rank}_past_label", "")
            score = row.get(f"top{rank}_score", "")
            if past == "":
                tops.append("")
            else:
                tops.append(f"{past} `{label}` ({score:.3f})")
        lines.append(
            f"| {row['step']} | `{row['step_label']}` | `{row['expected_refs']}` | {tops[0]} | {tops[1]} | {tops[2]} |"
        )

    output_md = Path(args.output_md)
    output_md.write_text("\n".join(lines) + "\n")
    print(json.dumps({"csv": str(output_csv), "md": str(output_md), "rows": len(rows)}))


if __name__ == "__main__":
    main()
