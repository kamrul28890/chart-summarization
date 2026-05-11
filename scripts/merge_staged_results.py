import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge staged part CSV files into final CSV/XLSX outputs."
    )
    parser.add_argument(
        "--run-dir",
        default="runs/qwen25vl_full",
        help="Run directory containing parts/.",
    )
    parser.add_argument(
        "--output-prefix",
        default="results/qwen25vl_full_latest",
        help="Output path without extension.",
    )
    parser.add_argument(
        "--done-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only rows with status=done.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    parts_dir = run_dir / "parts"
    part_paths = sorted(parts_dir.glob("part_*.csv"))
    if not part_paths:
        raise FileNotFoundError(f"No part CSV files found under {parts_dir}")

    frames = [pd.read_csv(path) for path in part_paths]
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["image_id"], keep="last")
    if args.done_only:
        merged = merged[merged["status"] == "done"].copy()
    merged = merged.sort_values(["split", "family", "numeric_id", "image_name"])

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    xlsx_path = output_prefix.with_suffix(".xlsx")
    merged.to_csv(csv_path, index=False, encoding="utf-8-sig")
    merged.to_excel(xlsx_path, index=False)

    print(f"Merged {len(part_paths)} part files")
    print(f"Rows: {len(merged)}")
    print(f"CSV:  {csv_path}")
    print(f"XLSX: {xlsx_path}")


if __name__ == "__main__":
    main()
