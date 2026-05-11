import argparse
import os
import zipfile
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)


VL_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
TRANS_MODEL_NAME = "facebook/nllb-200-distilled-600M"

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
        description="Run the ChartSumm Qwen2.5-VL desktop pipeline on a folder or ZIP of chart images."
    )
    parser.add_argument(
        "--input",
        default="AntuToDo10-20260511T040828Z-3-001.zip",
        help="Image folder or ZIP file. Images are searched recursively.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Folder for XLSX and partial CSV outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional image limit for smoke tests. Use 0 for all images.",
    )
    parser.add_argument(
        "--quantization",
        choices=["bnb4", "none"],
        default="bnb4",
        help="Default bnb4 uses 4-bit NF4 quantization for 8 GB GPUs.",
    )
    parser.add_argument(
        "--max-gpu-memory",
        default="5GiB",
        help="Per-GPU memory budget for accelerate device_map auto. Increase if VRAM is free.",
    )
    parser.add_argument(
        "--device-map",
        choices=["auto", "cuda"],
        default="auto",
        help="Use auto for GPU/CPU placement or cuda to force the full Qwen model onto GPU 0.",
    )
    parser.add_argument(
        "--translate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate Bangla summaries using NLLB after English summaries.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=250,
        help="Max tokens for each Qwen chart summary.",
    )
    return parser.parse_args()


def resolve_input(input_path: str) -> Path:
    path = Path(input_path)
    if path.is_dir():
        return path
    if not path.is_file() or path.suffix.lower() != ".zip":
        raise FileNotFoundError(f"Input must be an image folder or ZIP file: {path}")

    extract_root = Path("data") / "extracted" / path.stem
    marker = extract_root / ".extracted"
    if marker.exists():
        return extract_root

    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "r") as archive:
        archive.extractall(extract_root)
    marker.write_text("ok\n", encoding="utf-8")
    return extract_root


def find_images(folder: Path, limit: int = 0) -> list[Path]:
    image_exts = {".png", ".jpg", ".jpeg"}
    images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in image_exts)
    if limit > 0:
        images = images[:limit]
    if not images:
        raise RuntimeError(f"No images found under {folder}")
    return images


def load_qwen_model(quantization: str, max_gpu_memory: str, device_map: str):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this Python environment. Install a CUDA PyTorch wheel first."
        )

    processor = AutoProcessor.from_pretrained(
        VL_MODEL_NAME,
        min_pixels=256 * 28 * 28,
        max_pixels=1024 * 28 * 28,
    )

    kwargs = {
        "torch_dtype": torch.float16,
        "attn_implementation": "sdpa",
    }

    if device_map == "cuda":
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = {0: max_gpu_memory, "cpu": "40GiB"}

    if quantization == "bnb4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(VL_MODEL_NAME, **kwargs)
    model.eval()
    return processor, model


def load_translation_model():
    tokenizer = AutoTokenizer.from_pretrained(TRANS_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANS_MODEL_NAME)
    model.eval()
    return tokenizer, model


def generate_summary(processor, model, image: Image.Image, max_new_tokens: int) -> str:
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


def translate_en_to_bn(tokenizer, model, text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("ben_Beng"),
            max_new_tokens=256,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    args = parse_args()
    input_root = resolve_input(args.input)
    image_paths = find_images(input_root, args.limit)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(image_paths)} images under {input_root}")
    print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available'}")
    print(f"Qwen quantization: {args.quantization}")
    print(f"Qwen device map: {args.device_map}")

    processor, qwen_model = load_qwen_model(args.quantization, args.max_gpu_memory, args.device_map)
    trans_tokenizer = trans_model = None
    if args.translate:
        trans_tokenizer, trans_model = load_translation_model()

    rows = []
    partial_csv = output_dir / "partial_summaries.csv"

    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] {image_path.name}")
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            english_summary = generate_summary(
                processor, qwen_model, image, max_new_tokens=args.max_new_tokens
            )

        bangla_summary = ""
        if args.translate:
            bangla_summary = translate_en_to_bn(trans_tokenizer, trans_model, english_summary)

        rows.append(
            {
                "image_name": image_path.name,
                "image_path": str(image_path),
                "english_summary": english_summary,
                "bangla_summary": bangla_summary,
            }
        )
        pd.DataFrame(rows).to_csv(partial_csv, index=False, encoding="utf-8-sig")

    final_dataset = pd.DataFrame(rows)
    output_file = output_dir / "testset_summaries_1.xlsx"
    counter = 1
    while output_file.exists():
        counter += 1
        output_file = output_dir / f"testset_summaries_{counter}.xlsx"
    final_dataset.to_excel(output_file, index=False)
    print(f"Saved {output_file}")
    print(f"Partial CSV: {partial_csv}")


if __name__ == "__main__":
    main()
