# Chart Summarization with Qwen2.5-VL

This repository contains a local research pipeline for generating natural-language summaries of chart images. The current implementation uses `Qwen/Qwen2.5-VL-7B-Instruct` for visual chart understanding and `facebook/nllb-200-distilled-600M` for optional English-to-Bangla translation.

In simple terms: the system takes chart images, asks a vision-language model to explain the main pattern in each chart, optionally translates that explanation into Bangla, and saves the results as CSV/XLSX files for review.

## What Has Been Done

- Converted the original Colab notebook workflow into a repeatable desktop Python runner: `run_pipeline.py`.
- Added a CUDA-friendly configuration for an 8 GB RTX 3070 using 4-bit NF4 quantization.
- Ran a successful 200-image end-to-end test with reduced image-token budget to fit within available VRAM.
- Saved the completed run outputs under `results/`.
- Kept the full local dataset out of Git because it is about 5.3 GB and too large for normal GitHub repository storage.
- Documented setup, execution, directory structure, technical design, and future work.

## Current Results

The latest completed run processed 200 chart images using:

- Model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Quantization: 4-bit NF4 via bitsandbytes
- Image token budget: `--qwen-max-pixels 401408`
- Translation: enabled with NLLB
- Runtime: 166.8 minutes
- Average speed: 49.4 seconds/image

Published result files:

- `results/qwen2_5_vl_200_chart_summaries.csv`
- `results/qwen2_5_vl_200_chart_summaries.xlsx`

## Directory Structure

```text
.
|-- README.md
|-- requirements.txt
|-- run_pipeline.py
|-- run_pipeline_fp16.py
|-- datasets/
|   |-- README.md
|   `-- sample/
|       `-- antu_todo_200_charts.zip
|-- docs/
|   |-- desktop_cuda_runbook.md
|   |-- project_status.md
|   `-- technical_outline.md
|-- notebooks/
|   `-- Uthpol_DataSet_Pipeline_Qwen2_5_VL_7B_Instruct.ipynb
`-- results/
    |-- README.md
    |-- qwen2_5_vl_200_chart_summaries.csv
    `-- qwen2_5_vl_200_chart_summaries.xlsx
```

Generated working folders are intentionally ignored:

- `data/`: extracted ZIP contents used during local runs
- `outputs/`: scratch outputs from local runs
- `.venv/`: local Python environment
- `chart images/` and `chart images.zip`: full local dataset copy, not committed

## How The Pipeline Works

1. The input can be a folder or ZIP of chart images.
2. ZIP inputs are extracted under `data/extracted/`.
3. Images are found recursively.
4. Qwen2.5-VL receives each chart image plus a structured chart-analysis prompt.
5. The model returns one English paragraph per chart.
6. If translation is enabled, NLLB translates the English summary into Bangla.
7. The pipeline writes a partial CSV after each image and a final XLSX at the end.

## Quick Start

Create and activate a Python environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Verify CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Run a small smoke test:

```powershell
python run_pipeline.py --limit 2 --no-translate --qwen-max-pixels 401408
```

Run the included 200-image sample:

```powershell
python run_pipeline.py --qwen-max-pixels 401408
```

Use a full local dataset folder:

```powershell
python run_pipeline.py --input "chart images" --qwen-max-pixels 401408
```

## Future Work

- Run the pipeline on the full local dataset in batches.
- Add resume support so interrupted full-dataset runs skip already summarized images.
- Add automatic quality checks for empty, repeated, or malformed summaries.
- Compare Qwen2.5-VL results against smaller or faster chart-understanding models.
- Add evaluation metrics using a labeled reference subset.
- Package a reproducible experiment manifest with model version, prompt, runtime settings, and hardware details.

See `docs/project_status.md` and `docs/technical_outline.md` for a fuller research and engineering overview.
For full-dataset staged processing, see `docs/staged_workflow.md`.
