import argparse
import csv
import json
import math
import zipfile
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
PACKAGE_NAME = "cluster_qwen25vl_7b_fp16_nllb"
NOTEBOOK_NAME = "run_qwen25vl_7b_fp16_nllb_cluster.ipynb"


PROMPT = """You are an expert data analyst and data journalist.

Analyze the chart image carefully.

Think silently through these steps (DO NOT output them):
1. Identify chart type, domain, axes, units, and time range (if present)
2. Extract key data insights (trends, comparisons, patterns, anomalies)
3. Interpret domain meaning (causes, implications, real-world impact)

Then produce ONLY the final answer:

Summary:
Write a single, well-structured paragraph (4-6 sentences) that:
- Starts with the main trend or takeaway
- Includes at least one comparison or pattern
- Mentions any anomaly or notable feature (if present)
- Explains the real-world significance

Use clear, confident, natural English. No bullet points. No extra text."""


def parse_name(path: Path) -> tuple[str, str, int | None]:
    parts = path.stem.split("_")
    split = parts[0] if parts else ""
    family = parts[1] if len(parts) >= 3 else ""
    numeric_id = int(parts[-1]) if parts and parts[-1].isdigit() else None
    return split, family, numeric_id


def image_record(path: Path, image_id: str, part_id: int, root: Path) -> dict:
    split, family, numeric_id = parse_name(path)
    with Image.open(path) as image:
        width, height = image.size
        image_format = image.format or ""
    return {
        "part_id": part_id,
        "image_id": image_id,
        "image_name": path.name,
        "zip_relative_path": f"images/{path.name}",
        "source_relative_path": path.relative_to(root).as_posix(),
        "split": split,
        "family": family,
        "numeric_id": numeric_id if numeric_id is not None else "",
        "extension": path.suffix.lower(),
        "size_bytes": path.stat().st_size,
        "width": width,
        "height": height,
        "image_format": image_format,
    }


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def split_evenly(items: list[Path], parts: int) -> list[list[Path]]:
    base = len(items) // parts
    extra = len(items) % parts
    shards = []
    start = 0
    for index in range(parts):
        count = base + (1 if index < extra else 0)
        shards.append(items[start : start + count])
        start += count
    return shards


def md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(True),
    }


def write_notebook(path: Path):
    cells = [
        md_cell(
            "# Qwen2.5-VL 7B FP16 Chart Summarization + NLLB Bangla Translation\n"
            "\n"
            "This notebook is prepared for a cluster/Jupyter environment with internet access. "
            "It follows the original Colab-style flow: setup, device check, Model 1 "
            "(Qwen2.5-VL), Model 2 (NLLB), dataset loading, inference loop, and output saving.\n"
            "\n"
            "The dataset has been split into 10 ZIP files. Run one part at a time for normal "
            "cluster jobs, or enable the all-parts loop if the session has enough time."
        ),
        md_cell(
            "## Phase 0 - What to upload\n"
            "\n"
            "Upload this whole folder to the cluster:\n"
            "\n"
            "- `run_qwen25vl_7b_fp16_nllb_cluster.ipynb`\n"
            "- `dataset_parts/part_01.zip` through `dataset_parts/part_10.zip`\n"
            "- `manifests/` folder\n"
            "- `merge_cluster_outputs.py`\n"
            "\n"
            "Open this notebook from the uploaded folder so all relative paths work."
        ),
        md_cell("## Phase 1 - Install dependencies"),
        code_cell(
            "# Run this cell once per fresh cluster environment.\n"
            "# If your cluster already has CUDA PyTorch installed, you can skip the torch install line.\n"
            "%pip install -q --upgrade pip\n"
            "%pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu126\n"
            "%pip install -q \"transformers>=4.49\" \"accelerate>=1.0\" pandas openpyxl pillow tqdm\n"
        ),
        md_cell("## Phase 2 - Imports and device check"),
        code_cell(
            "import gc\n"
            "import os\n"
            "import time\n"
            "import zipfile\n"
            "from pathlib import Path\n"
            "\n"
            "import pandas as pd\n"
            "import torch\n"
            "from PIL import Image\n"
            "from tqdm.auto import tqdm\n"
            "from transformers import AutoModelForSeq2SeqLM, AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration\n"
            "\n"
            "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n"
            "print(f\"Using device: {device}\")\n"
            "if torch.cuda.is_available():\n"
            "    print(torch.__version__)\n"
            "    print(torch.cuda.get_device_name(0))\n"
            "    free, total = torch.cuda.mem_get_info(0)\n"
            "    print(f\"CUDA memory: {free / 1024**3:.2f} GiB free / {total / 1024**3:.2f} GiB total\")\n"
        ),
        md_cell("## Phase 3 - Run configuration"),
        code_cell(
            "# Normal use: keep RUN_ALL_PARTS = False and set PART_ID to 1..10.\n"
            "# Long session use: set RUN_ALL_PARTS = True to process every part in order.\n"
            "PART_ID = 1\n"
            "RUN_ALL_PARTS = False\n"
            "\n"
            "DATASET_ZIP_DIR = Path(\"dataset_parts\")\n"
            "WORK_DIR = Path(\"cluster_work\")\n"
            "OUTPUT_DIR = Path(\"cluster_outputs\")\n"
            "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
            "\n"
            "VL_MODEL_NAME = \"Qwen/Qwen2.5-VL-7B-Instruct\"\n"
            "TRANS_MODEL_NAME = \"facebook/nllb-200-distilled-600M\"\n"
            "\n"
            "# FP16 7B usually needs a larger cluster GPU than an 8 GB desktop card.\n"
            "QWEN_MIN_PIXELS = 128 * 28 * 28\n"
            "QWEN_MAX_PIXELS = 768 * 28 * 28\n"
            "MAX_NEW_TOKENS_QWEN = 250\n"
            "MAX_NEW_TOKENS_NLLB = 256\n"
            "\n"
            "# Keep NLLB on CPU by default so the FP16 Qwen model owns GPU memory.\n"
            "# Change to \"cuda\" only if the GPU has comfortable spare memory.\n"
            "TRANSLATION_DEVICE = \"cpu\"\n"
            "\n"
            "IMAGE_EXTS = {\".png\", \".jpg\", \".jpeg\"}\n"
            "parts_to_run = list(range(1, 11)) if RUN_ALL_PARTS else [PART_ID]\n"
            "parts_to_run\n"
        ),
        md_cell("# Model - 1 : Chart Summary Qwen2.5-VL-7B-Instruct"),
        code_cell(
            "if device != \"cuda\":\n"
            "    raise RuntimeError(\"CUDA is required for the FP16 Qwen2.5-VL 7B run.\")\n"
            "\n"
            "qwen_processor = AutoProcessor.from_pretrained(\n"
            "    VL_MODEL_NAME,\n"
            "    min_pixels=QWEN_MIN_PIXELS,\n"
            "    max_pixels=QWEN_MAX_PIXELS,\n"
            ")\n"
            "\n"
            "if \"qwen_model\" not in globals() or qwen_model is None:\n"
            "    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(\n"
            "        VL_MODEL_NAME,\n"
            "        torch_dtype=torch.float16,\n"
            "        attn_implementation=\"sdpa\",\n"
            "        device_map={\"\": 0},\n"
            "    )\n"
            "    qwen_model.eval()\n"
            "\n"
            "print(\"Qwen2.5-VL 7B FP16 loaded.\")\n"
        ),
        md_cell("# Model - 2: Translation Model NLLB200\n\n`facebook/nllb-200-distilled-600M`"),
        code_cell(
            "if \"trans_tokenizer\" not in globals() or trans_tokenizer is None:\n"
            "    trans_tokenizer = AutoTokenizer.from_pretrained(TRANS_MODEL_NAME)\n"
            "\n"
            "if \"trans_model\" not in globals() or trans_model is None:\n"
            "    trans_model = AutoModelForSeq2SeqLM.from_pretrained(TRANS_MODEL_NAME)\n"
            "    trans_model.to(TRANSLATION_DEVICE)\n"
            "    trans_model.eval()\n"
            "\n"
            "print(f\"NLLB translation model loaded on {TRANSLATION_DEVICE}.\")\n"
        ),
        md_cell("## Pipeline: Planner + Insight Extractor + Summarizer = Qwen2.5-VL-7B-Instruct"),
        code_cell(
            f"SINGLE_PASS_PROMPT = {PROMPT!r}\n"
            "\n"
            "def generate_chart_summary(image, max_new_tokens=MAX_NEW_TOKENS_QWEN):\n"
            "    messages = [\n"
            "        {\n"
            "            \"role\": \"user\",\n"
            "            \"content\": [\n"
            "                {\"type\": \"image\", \"image\": image},\n"
            "                {\"type\": \"text\", \"text\": SINGLE_PASS_PROMPT},\n"
            "            ],\n"
            "        }\n"
            "    ]\n"
            "    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n"
            "    inputs = qwen_processor(text=[text], images=[image], return_tensors=\"pt\")\n"
            "    inputs = {key: value.to(\"cuda\") for key, value in inputs.items()}\n"
            "    input_len = inputs[\"input_ids\"].shape[1]\n"
            "\n"
            "    with torch.inference_mode():\n"
            "        output = qwen_model.generate(\n"
            "            **inputs,\n"
            "            max_new_tokens=max_new_tokens,\n"
            "            do_sample=False,\n"
            "            use_cache=True,\n"
            "            temperature=None,\n"
            "            top_p=None,\n"
            "        )\n"
            "\n"
            "    new_tokens = output[0][input_len:]\n"
            "    return qwen_processor.decode(new_tokens, skip_special_tokens=True).strip()\n"
            "\n"
            "\n"
            "def translate_en_to_bn(text, max_new_tokens=MAX_NEW_TOKENS_NLLB):\n"
            "    inputs = trans_tokenizer(text, return_tensors=\"pt\", truncation=True, max_length=512)\n"
            "    inputs = {key: value.to(TRANSLATION_DEVICE) for key, value in inputs.items()}\n"
            "    with torch.inference_mode():\n"
            "        outputs = trans_model.generate(\n"
            "            **inputs,\n"
            "            forced_bos_token_id=trans_tokenizer.convert_tokens_to_ids(\"ben_Beng\"),\n"
            "            max_new_tokens=max_new_tokens,\n"
            "        )\n"
            "    return trans_tokenizer.decode(outputs[0], skip_special_tokens=True)\n"
        ),
        md_cell("# Load Split Dataset"),
        code_cell(
            "def extract_part(part_id):\n"
            "    zip_path = DATASET_ZIP_DIR / f\"part_{part_id:02d}.zip\"\n"
            "    if not zip_path.exists():\n"
            "        raise FileNotFoundError(f\"Missing dataset ZIP: {zip_path}\")\n"
            "    extract_dir = WORK_DIR / f\"part_{part_id:02d}\"\n"
            "    marker = extract_dir / \".extracted\"\n"
            "    if not marker.exists():\n"
            "        extract_dir.mkdir(parents=True, exist_ok=True)\n"
            "        with zipfile.ZipFile(zip_path, \"r\") as archive:\n"
            "            archive.extractall(extract_dir)\n"
            "        marker.write_text(\"ok\\n\", encoding=\"utf-8\")\n"
            "    image_files = sorted(path for path in extract_dir.rglob(\"*\") if path.suffix.lower() in IMAGE_EXTS)\n"
            "    if not image_files:\n"
            "        raise RuntimeError(f\"No images found after extracting {zip_path}\")\n"
            "    return extract_dir, image_files\n"
            "\n"
            "\n"
            "def metadata_from_name(path):\n"
            "    parts = path.stem.split(\"_\")\n"
            "    split = parts[0] if parts else \"\"\n"
            "    family = parts[1] if len(parts) >= 3 else \"\"\n"
            "    numeric_id = int(parts[-1]) if parts and parts[-1].isdigit() else None\n"
            "    image_id = f\"{split}_{family}_{numeric_id}\" if split and family and numeric_id is not None else path.stem\n"
            "    return image_id, split, family, numeric_id\n"
            "\n"
            "\n"
            "extract_dir, image_files = extract_part(parts_to_run[0])\n"
            "df_images = pd.DataFrame({\"image\": [str(path) for path in image_files]})\n"
            "df_images[\"image_name\"] = df_images[\"image\"].apply(lambda x: Path(x).name)\n"
            "print(f\"Preview part {parts_to_run[0]}: {len(df_images)} images\")\n"
            "df_images.head()\n"
        ),
        md_cell("## New Dataset Creation by Single Prompt technique"),
        code_cell(
            "def process_part(part_id):\n"
            "    extract_dir, image_files = extract_part(part_id)\n"
            "    output_csv = OUTPUT_DIR / f\"part_{part_id:02d}_qwen25vl7b_fp16_nllb.csv\"\n"
            "    output_xlsx = OUTPUT_DIR / f\"part_{part_id:02d}_qwen25vl7b_fp16_nllb.xlsx\"\n"
            "\n"
            "    completed = set()\n"
            "    rows = []\n"
            "    if output_csv.exists():\n"
            "        previous = pd.read_csv(output_csv)\n"
            "        rows = previous.to_dict(\"records\")\n"
            "        completed = set(previous.loc[previous[\"status\"] == \"done\", \"image_id\"].astype(str))\n"
            "        print(f\"Part {part_id:02d}: resuming with {len(completed)} completed rows.\")\n"
            "\n"
            "    for image_path in tqdm(image_files, desc=f\"Part {part_id:02d}\", unit=\"img\"):\n"
            "        image_id, split, family, numeric_id = metadata_from_name(image_path)\n"
            "        if image_id in completed:\n"
            "            continue\n"
            "\n"
            "        started = time.perf_counter()\n"
            "        row = {\n"
            "            \"part_id\": part_id,\n"
            "            \"image_id\": image_id,\n"
            "            \"image_name\": image_path.name,\n"
            "            \"image_path\": str(image_path),\n"
            "            \"split\": split,\n"
            "            \"family\": family,\n"
            "            \"numeric_id\": numeric_id if numeric_id is not None else \"\",\n"
            "            \"english_summary\": \"\",\n"
            "            \"bangla_summary\": \"\",\n"
            "            \"status\": \"done\",\n"
            "            \"error\": \"\",\n"
            "            \"seconds\": \"\",\n"
            "        }\n"
            "\n"
            "        try:\n"
            "            with Image.open(image_path) as img:\n"
            "                image = img.convert(\"RGB\")\n"
            "                english_summary = generate_chart_summary(image)\n"
            "            bangla_summary = translate_en_to_bn(english_summary)\n"
            "            row[\"english_summary\"] = english_summary\n"
            "            row[\"bangla_summary\"] = bangla_summary\n"
            "        except Exception as exc:\n"
            "            row[\"status\"] = \"failed\"\n"
            "            row[\"error\"] = f\"{type(exc).__name__}: {exc}\"\n"
            "        finally:\n"
            "            row[\"seconds\"] = round(time.perf_counter() - started, 2)\n"
            "            rows.append(row)\n"
            "            pd.DataFrame(rows).drop_duplicates(\"image_id\", keep=\"last\").to_csv(output_csv, index=False, encoding=\"utf-8-sig\")\n"
            "\n"
            "            if torch.cuda.is_available():\n"
            "                torch.cuda.empty_cache()\n"
            "            gc.collect()\n"
            "\n"
            "    final_df = pd.DataFrame(rows).drop_duplicates(\"image_id\", keep=\"last\")\n"
            "    final_df.to_csv(output_csv, index=False, encoding=\"utf-8-sig\")\n"
            "    final_df.to_excel(output_xlsx, index=False)\n"
            "    print(f\"Saved part {part_id:02d}: {output_csv}\")\n"
            "    print(f\"Saved part {part_id:02d}: {output_xlsx}\")\n"
            "    return final_df\n"
        ),
        md_cell("# Create Final Dataset & Save"),
        code_cell(
            "part_frames = []\n"
            "for part_id in parts_to_run:\n"
            "    part_frames.append(process_part(part_id))\n"
            "\n"
            "final_dataset = pd.concat(part_frames, ignore_index=True) if part_frames else pd.DataFrame()\n"
            "combined_csv = OUTPUT_DIR / \"combined_current_session_qwen25vl7b_fp16_nllb.csv\"\n"
            "combined_xlsx = OUTPUT_DIR / \"combined_current_session_qwen25vl7b_fp16_nllb.xlsx\"\n"
            "final_dataset.to_csv(combined_csv, index=False, encoding=\"utf-8-sig\")\n"
            "final_dataset.to_excel(combined_xlsx, index=False)\n"
            "print(\"Final dataset created for this notebook session:\")\n"
            "print(combined_csv)\n"
            "print(combined_xlsx)\n"
            "final_dataset.head()\n"
        ),
        md_cell(
            "## Phase 7 - After all parts finish\n"
            "\n"
            "Download or keep all `cluster_outputs/part_*.csv` files together. Then run:\n"
            "\n"
            "```bash\n"
            "python merge_cluster_outputs.py --input-dir cluster_outputs --output-prefix cluster_outputs/qwen25vl7b_fp16_nllb_full_dataset\n"
            "```\n"
            "\n"
            "That creates the full merged CSV and XLSX."
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def write_merge_script(path: Path):
    path.write_text(
        """import argparse
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
""",
        encoding="utf-8",
    )


def write_readme(path: Path, total_images: int, part_counts: list[int]):
    counts = "\n".join(
        f"- `part_{index:02d}.zip`: {count:,} images"
        for index, count in enumerate(part_counts, start=1)
    )
    path.write_text(
        f"""# Cluster Qwen2.5-VL 7B FP16 + NLLB Package

This folder is ready to upload to a Jupyter cluster environment.

## Contents

- `{NOTEBOOK_NAME}`: Colab-style notebook for Qwen2.5-VL 7B FP16 chart summarization plus NLLB Bangla translation.
- `dataset_parts/part_01.zip` through `dataset_parts/part_10.zip`: full dataset split into 10 nearly equal ZIP shards.
- `manifests/`: per-part manifests and a full manifest for auditing.
- `merge_cluster_outputs.py`: merges part CSV outputs into one final CSV/XLSX.

## Dataset split

Total images: {total_images:,}

{counts}

## How to run

1. Upload this whole folder to the cluster.
2. Open `{NOTEBOOK_NAME}` from this folder.
3. Run the install/import/model cells.
4. In the configuration cell, keep `RUN_ALL_PARTS = False` and set `PART_ID = 1` through `10` for normal jobs.
5. If the cluster session has enough time, set `RUN_ALL_PARTS = True` to process all parts sequentially.
6. Outputs are written into `cluster_outputs/` after every image, so interrupted jobs can resume from the part CSV.
7. After all parts finish, run:

```bash
python merge_cluster_outputs.py --input-dir cluster_outputs --output-prefix cluster_outputs/qwen25vl7b_fp16_nllb_full_dataset
```

## Notes

- The notebook intentionally loads `Qwen/Qwen2.5-VL-7B-Instruct` in FP16, not 4-bit.
- NLLB is kept on CPU by default so Qwen can use the GPU memory. Change `TRANSLATION_DEVICE = "cuda"` only if the cluster GPU has spare memory.
- The notebook assumes internet access for downloading Hugging Face models.
""",
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare cluster notebook and 10 dataset ZIP shards.")
    parser.add_argument("--input", default="chart images", help="Full local chart image folder.")
    parser.add_argument("--output", default=PACKAGE_NAME, help="Output package folder.")
    parser.add_argument("--parts", type=int, default=10, help="Number of dataset shards.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_root = Path(args.input)
    output_root = Path(args.output)
    parts_dir = output_root / "dataset_parts"
    manifests_dir = output_root / "manifests"
    parts_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(path for path in input_root.rglob("*") if path.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise RuntimeError(f"No images found under {input_root}")

    shards = split_evenly(images, args.parts)
    all_rows = []
    part_counts = []
    for part_index, shard in enumerate(shards, start=1):
        part_counts.append(len(shard))
        rows = []
        zip_path = parts_dir / f"part_{part_index:02d}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
            for image_path in tqdm(shard, desc=f"part_{part_index:02d}", unit="img"):
                split, family, numeric_id = parse_name(image_path)
                image_id = (
                    f"{split}_{family}_{numeric_id}"
                    if numeric_id is not None and split and family
                    else image_path.stem
                )
                archive.write(image_path, arcname=f"images/{image_path.name}")
                row = image_record(image_path, image_id, part_index, input_root)
                rows.append(row)
                all_rows.append(row)
        write_csv(manifests_dir / f"part_{part_index:02d}_manifest.csv", rows)

    write_csv(manifests_dir / "all_parts_manifest.csv", all_rows)
    write_notebook(output_root / NOTEBOOK_NAME)
    write_merge_script(output_root / "merge_cluster_outputs.py")
    write_readme(output_root / "README.md", len(images), part_counts)

    summary = {
        "package": str(output_root),
        "total_images": len(images),
        "parts": args.parts,
        "part_counts": part_counts,
        "notebook": str(output_root / NOTEBOOK_NAME),
    }
    (output_root / "package_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
