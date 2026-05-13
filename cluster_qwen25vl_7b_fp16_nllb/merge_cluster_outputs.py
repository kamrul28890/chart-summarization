import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Merge cluster part outputs.")
    parser.add_argument("--input-dir", default="cluster_outputs")
    parser.add_argument("--output-prefix", default="cluster_outputs/qwen25vl7b_fp16_nllb_full_dataset")
    parser.add_argument("--done-only", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    part_files = sorted(input_dir.glob("part_*_qwen25vl7b_fp16_nllb.csv"))
    if not part_files:
        raise FileNotFoundError(f"No part CSV files found in {input_dir}")

    frames = [pd.read_csv(path) for path in part_files]
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["image_id"], keep="last")
    if args.done_only:
        merged = merged[merged["status"] == "done"].copy()
    merged = merged.sort_values(["part_id", "split", "family", "numeric_id", "image_name"])

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    xlsx_path = output_prefix.with_suffix(".xlsx")
    merged.to_csv(csv_path, index=False, encoding="utf-8-sig")
    merged.to_excel(xlsx_path, index=False)

    print(f"Merged files: {len(part_files)}")
    print(f"Rows: {len(merged)}")
    print(f"CSV: {csv_path}")
    print(f"XLSX: {xlsx_path}")


if __name__ == "__main__":
    main()
