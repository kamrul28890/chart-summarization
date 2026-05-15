from __future__ import annotations

import argparse
import csv
import gc
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter


VL_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
TRANS_MODEL_NAME = "facebook/nllb-200-distilled-600M"
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

RESULT_FIELDS = [
    "image_id",
    "image_name",
    "image_path",
    "source_zip",
    "split",
    "family",
    "numeric_id",
    "english_summary",
    "bangla_summary",
    "status",
    "error",
    "seconds",
    "completed_at",
]

SINGLE_PASS_PROMPT = """You are an expert data analyst and data journalist.

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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Cluster-friendly one-file runner for Qwen2.5-VL 7B FP16 chart "
            "summarization plus NLLB Bangla translation. Writes one resumable CSV."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=["dataset_parts"],
        help=(
            "Input image folder(s), ZIP file(s), or folder(s) containing ZIP files. "
            "Default: dataset_parts"
        ),
    )
    parser.add_argument(
        "--output-csv",
        default="cluster_outputs/qwen25vl7b_fp16_nllb_all_outputs.csv",
        help="Single output CSV. Existing done rows are skipped unless --overwrite is used.",
    )
    parser.add_argument(
        "--work-dir",
        default="cluster_work",
        help="Folder for extracted ZIP contents.",
    )
    parser.add_argument(
        "--vl-model-name",
        default=VL_MODEL_NAME,
        help="Qwen vision-language model name or local path.",
    )
    parser.add_argument(
        "--translation-model-name",
        default=TRANS_MODEL_NAME,
        help="NLLB translation model name or local path.",
    )
    parser.add_argument(
        "--qwen-min-pixels",
        type=int,
        default=128 * 28 * 28,
        help="Minimum visual token pixel budget.",
    )
    parser.add_argument(
        "--qwen-max-pixels",
        type=int,
        default=768 * 28 * 28,
        help="Maximum visual token pixel budget. Lower this if the cluster GPU runs out of memory.",
    )
    parser.add_argument(
        "--max-new-tokens-qwen",
        type=int,
        default=250,
        help="Maximum new tokens for each English chart summary.",
    )
    parser.add_argument(
        "--max-new-tokens-nllb",
        type=int,
        default=256,
        help="Maximum new tokens for each Bangla translation.",
    )
    parser.add_argument(
        "--translation-device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for NLLB. Keep CPU if Qwen needs all GPU memory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional image limit for smoke tests. Use 0 for all images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Start a fresh output CSV instead of resuming.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry rows already present with status=failed.",
    )
    parser.add_argument(
        "--no-translate",
        action="store_true",
        help="Skip NLLB translation and leave bangla_summary empty.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_name(path: Path) -> tuple[str, str, int | None]:
    parts = path.stem.split("_")
    split = parts[0] if parts else ""
    family = parts[1] if len(parts) >= 3 else ""
    numeric_id = int(parts[-1]) if parts and parts[-1].isdigit() else None
    return split, family, numeric_id


def image_id_from_path(path: Path) -> str:
    split, family, numeric_id = parse_name(path)
    if split and family and numeric_id is not None:
        return f"{split}_{family}_{numeric_id}"
    return path.stem


def extract_zip(zip_path: Path, work_dir: Path) -> Path:
    extract_dir = work_dir / zip_path.stem
    marker = extract_dir / ".extracted"
    if marker.exists():
        return extract_dir

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_dir)
    marker.write_text("ok\n", encoding="utf-8")
    return extract_dir


def find_images(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTS)


def discover_images(input_paths: list[str], work_dir: Path, limit: int) -> list[dict]:
    records = []
    seen_paths = set()

    for input_value in input_paths:
        input_path = Path(input_value)
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")

        if input_path.is_file() and input_path.suffix.lower() == ".zip":
            extract_dir = extract_zip(input_path, work_dir)
            image_paths = find_images(extract_dir)
            source_zip = str(input_path)
        elif input_path.is_dir():
            image_paths = find_images(input_path)
            source_zip = ""
            zip_paths = sorted(path for path in input_path.rglob("*.zip"))
            for zip_path in zip_paths:
                extract_dir = extract_zip(zip_path, work_dir)
                for image_path in find_images(extract_dir):
                    key = image_path.resolve()
                    if key in seen_paths:
                        continue
                    seen_paths.add(key)
                    records.append(
                        {
                            "image_path": image_path,
                            "source_zip": str(zip_path),
                        }
                    )
        else:
            raise FileNotFoundError(f"Input must be an image folder or ZIP file: {input_path}")

        for image_path in image_paths:
            key = image_path.resolve()
            if key in seen_paths:
                continue
            seen_paths.add(key)
            records.append({"image_path": image_path, "source_zip": source_zip})

    records = sorted(records, key=lambda item: str(item["image_path"]))
    if limit > 0:
        records = records[:limit]
    if not records:
        raise RuntimeError(f"No images found in inputs: {input_paths}")
    return records


def load_completed(output_csv: Path, retry_failed: bool) -> set[str]:
    if not output_csv.exists():
        return set()

    skip_statuses = {"done"}
    if not retry_failed:
        skip_statuses.add("failed")

    completed = set()
    with output_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "image_id" not in reader.fieldnames or "status" not in reader.fieldnames:
            raise RuntimeError(f"Existing CSV has unexpected columns: {output_csv}")
        for row in reader:
            if row.get("status") in skip_statuses and row.get("image_id"):
                completed.add(str(row["image_id"]))
    return completed


def append_result(output_csv: Path, row: dict):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_csv.exists()
    with output_csv.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def format_cuda_memory() -> str:
    import torch

    if not torch.cuda.is_available():
        return "CUDA unavailable"
    free, total = torch.cuda.mem_get_info(0)
    return f"{free / 1024**3:.2f} GiB free / {total / 1024**3:.2f} GiB total"


def load_qwen(args):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Qwen2.5-VL 7B FP16 cluster run.")

    processor = AutoProcessor.from_pretrained(
        args.vl_model_name,
        min_pixels=args.qwen_min_pixels,
        max_pixels=args.qwen_max_pixels,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.vl_model_name,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    model.eval()
    return processor, model


def load_nllb(args):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.translation_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.translation_model_name)
    model.to(args.translation_device)
    model.eval()
    return tokenizer, model


def generate_summary(processor, model, image: Image.Image, max_new_tokens: int) -> str:
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": SINGLE_PASS_PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {key: value.to("cuda") for key, value in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            temperature=None,
            top_p=None,
        )

    new_tokens = output[0][input_len:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


def translate_en_to_bn(tokenizer, model, text: str, device: str, max_new_tokens: int) -> str:
    import torch

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("ben_Beng"),
            max_new_tokens=max_new_tokens,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    args = parse_args()
    output_csv = Path(args.output_csv)
    work_dir = Path(args.work_dir)

    if args.overwrite and output_csv.exists():
        output_csv.unlink()

    records = discover_images(args.input, work_dir, args.limit)
    completed = load_completed(output_csv, args.retry_failed)
    pending = [record for record in records if image_id_from_path(record["image_path"]) not in completed]

    print(f"Discovered images: {len(records)}")
    print(f"Already skipped from CSV: {len(records) - len(pending)}")
    print(f"Pending images: {len(pending)}")
    print(f"Output CSV: {output_csv}")

    import torch

    print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available'}")
    print(f"CUDA memory before loading Qwen: {format_cuda_memory()}")
    print(f"Qwen model: {args.vl_model_name}")
    print("Qwen precision: FP16")
    print(f"Qwen pixel budget: min={args.qwen_min_pixels}, max={args.qwen_max_pixels}")
    print(f"NLLB translation: {'disabled' if args.no_translate else args.translation_device}")

    if not pending:
        print("Nothing to process.")
        return

    processor, qwen_model = load_qwen(args)
    trans_tokenizer = trans_model = None
    if not args.no_translate:
        trans_tokenizer, trans_model = load_nllb(args)

    from PIL import Image
    from tqdm.auto import tqdm

    done_count = 0
    failed_count = 0
    progress = tqdm(pending, desc="Summarizing charts", unit="img")
    for record in progress:
        started = perf_counter()
        image_path = record["image_path"]
        image_id = image_id_from_path(image_path)
        split, family, numeric_id = parse_name(image_path)
        row = {
            "image_id": image_id,
            "image_name": image_path.name,
            "image_path": str(image_path),
            "source_zip": record["source_zip"],
            "split": split,
            "family": family,
            "numeric_id": numeric_id if numeric_id is not None else "",
            "english_summary": "",
            "bangla_summary": "",
            "status": "done",
            "error": "",
            "seconds": "",
            "completed_at": "",
        }

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
                english_summary = generate_summary(
                    processor,
                    qwen_model,
                    image,
                    max_new_tokens=args.max_new_tokens_qwen,
                )
            bangla_summary = ""
            if not args.no_translate:
                bangla_summary = translate_en_to_bn(
                    trans_tokenizer,
                    trans_model,
                    english_summary,
                    args.translation_device,
                    args.max_new_tokens_nllb,
                )
            row["english_summary"] = english_summary
            row["bangla_summary"] = bangla_summary
            done_count += 1
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            failed_count += 1
        finally:
            row["seconds"] = f"{perf_counter() - started:.2f}"
            row["completed_at"] = now_iso()
            append_result(output_csv, row)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            progress.set_postfix({"done": done_count, "failed": failed_count})

    print(f"Finished. Done: {done_count}. Failed: {failed_count}.")
    print(f"CSV: {output_csv}")


if __name__ == "__main__":
    main()
