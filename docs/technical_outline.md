# Technical Outline

## System Overview

The repository implements an image-to-summary pipeline for chart images.

```text
Input image folder or ZIP
        |
        v
Image discovery and preprocessing
        |
        v
Qwen2.5-VL chart understanding
        |
        v
English chart summary
        |
        v
Optional NLLB English-to-Bangla translation
        |
        v
CSV and XLSX outputs
```

## Main Components

### `run_pipeline.py`

The production entrypoint. It handles:

- command-line arguments
- ZIP extraction
- recursive image discovery
- Qwen2.5-VL loading
- optional 4-bit quantization
- per-image summary generation
- optional Bangla translation
- partial CSV writing
- final XLSX export
- basic runtime and ETA reporting

### `run_pipeline_fp16.py`

A small compatibility runner for testing unquantized FP16 model loading. On an 8 GB RTX 3070, this path may fail with CUDA out-of-memory and is mainly useful as a diagnostic comparison against the 4-bit path.

### `notebooks/`

Contains the original notebook implementation. It is preserved for provenance and comparison, while the script-based runner is the preferred workflow for repeatable local runs.

### `datasets/`

Contains the publishable sample ZIP. The full local dataset is documented but intentionally ignored.

### `results/`

Contains final, named outputs from completed runs. Scratch outputs remain under ignored `outputs/`.

## Model Choices

### Vision-Language Model

`Qwen/Qwen2.5-VL-7B-Instruct` is used because it can read chart images and produce structured text descriptions.

### Translation Model

`facebook/nllb-200-distilled-600M` is used for English-to-Bangla translation after the English chart summary is generated.

## GPU Strategy

The desktop GPU has 8 GB of VRAM and also drives the Windows display. Some VRAM is always occupied by desktop applications. To fit the model safely:

- Qwen is loaded with 4-bit NF4 quantization.
- The image-token budget can be reduced with `--qwen-max-pixels`.
- The successful 200-image run used `--qwen-max-pixels 401408`.
- Translation runs on CPU in the current implementation.

## Key Runtime Arguments

```text
--input             Folder or ZIP of chart images
--output-dir        Scratch output directory
--limit             Optional image limit for smoke tests
--quantization      bnb4 or none
--device-map        cuda or auto
--max-gpu-memory    GPU budget for auto placement
--qwen-min-pixels   Minimum image-token pixel budget
--qwen-max-pixels   Maximum image-token pixel budget
--translate         Enable or disable Bangla translation
--max-new-tokens    Generation length for chart summaries
```

## Data Flow Details

1. `resolve_input()` accepts either a folder or ZIP file.
2. ZIP files are extracted under `data/extracted/`.
3. `find_images()` recursively finds `.png`, `.jpg`, and `.jpeg` images.
4. `load_qwen_model()` initializes the processor and model.
5. `generate_summary()` constructs a multimodal chat prompt and decodes only newly generated tokens.
6. `translate_en_to_bn()` translates each English summary when translation is enabled.
7. Each processed row is immediately written to `outputs/partial_summaries.csv`.
8. At completion, the full table is exported to an XLSX file.

## Easy Explanation

The script is a batch worker. It opens each chart image, sends it to a chart-reading AI model, stores the English explanation, translates it if requested, and keeps saving progress so the work is not lost. The reduced image budget is the main setting that keeps the run practical on an 8 GB GPU.
