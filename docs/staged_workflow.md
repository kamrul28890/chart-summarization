# Staged Full-Dataset Workflow

The full local dataset is too large to process comfortably in one desktop run. The staged workflow processes a fixed number of images per command, records progress in SQLite, and resumes from the last unfinished point.

## Why This Workflow

Long-running chart summarization can be interrupted by GPU memory pressure, Windows restarts, power loss, or model errors. A staged workflow avoids losing progress:

- one manifest defines the full dataset
- one SQLite database tracks pending, running, done, and failed rows
- one part CSV is written per stage
- every processed image is saved immediately
- failed rows are isolated instead of stopping the full run
- interrupted `running` rows are requeued on the next command

## Step 1: Build The Manifest

Run this once:

```powershell
python scripts/build_manifest.py --input "chart images" --output manifests/full_dataset_manifest.csv
```

This creates one row per image with:

- image ID
- filename and path
- split: `train`, `valid`, or `test`
- family: `k` or `s`
- numeric ID
- file size
- dimensions
- SHA-256 hash

For a quick smoke test:

```powershell
python scripts/build_manifest.py --input "chart images" --output manifests/smoke_manifest.csv --limit 20
```

## Step 2: Preview The Next Stage

Preview the next 500 images without loading Qwen:

```powershell
python run_staged_pipeline.py --stage-size 500 --dry-run
```

Check progress at any time:

```powershell
python run_staged_pipeline.py --status
```

## Step 3: Process One Stage

Run one 500-image stage:

```powershell
python run_staged_pipeline.py --stage-size 500 --qwen-max-pixels 401408
```

Run the same command again to process the next 500 pending images. The state database decides what comes next.

## Recommended Processing Order

Start with validation and test splits before training splits:

```powershell
python run_staged_pipeline.py --split valid --stage-size 500 --qwen-max-pixels 401408
python run_staged_pipeline.py --split test --stage-size 500 --qwen-max-pixels 401408
python run_staged_pipeline.py --split train --stage-size 500 --qwen-max-pixels 401408
```

Use `--family k` or `--family s` if you want to process one source family at a time:

```powershell
python run_staged_pipeline.py --split valid --family k --stage-size 500 --qwen-max-pixels 401408
```

## Where Stage Outputs Go

```text
runs/
`-- qwen25vl_full/
    |-- run_config.json
    |-- state.sqlite
    `-- parts/
        |-- part_000001.csv
        |-- part_000002.csv
        `-- ...
```

These folders are ignored by Git because they are local run artifacts.

## Step 4: Merge Results

Merge completed part files into final CSV/XLSX files:

```powershell
python scripts/merge_staged_results.py --run-dir runs/qwen25vl_full --output-prefix results/qwen25vl_full_latest
```

This writes:

```text
results/qwen25vl_full_latest.csv
results/qwen25vl_full_latest.xlsx
```

## Retrying Failed Rows

After checking failures, retry them with:

```powershell
python run_staged_pipeline.py --retry-failed --stage-size 500 --qwen-max-pixels 401408
```

## Practical Desktop Settings

For the current RTX 3070 8 GB setup, use:

```powershell
python run_staged_pipeline.py --stage-size 500 --qwen-max-pixels 401408
```

## Qwen2.5-VL 3B FP16 Run

For a smaller Qwen vision-language model without 4-bit quantization, use the 3B FP16 wrapper:

```powershell
python run_staged_qwen3b_fp16.py --split valid --stage-size 500
```

This is equivalent to:

```powershell
python run_staged_pipeline.py `
  --run-id qwen25vl_3b_fp16 `
  --vl-model-name Qwen/Qwen2.5-VL-3B-Instruct `
  --quantization none `
  --device-map cuda `
  --split valid `
  --stage-size 500 `
  --qwen-max-pixels 401408
```

Use this run ID separately from the default 7B quantized run so the result parts and state database do not mix.

If you hit CUDA out-of-memory, lower the image budget:

```powershell
python run_staged_pipeline.py --stage-size 500 --qwen-max-pixels 301056
```

If you want faster dry checks without translation:

```powershell
python run_staged_pipeline.py --stage-size 500 --no-translate --qwen-max-pixels 401408
```

## What This Is And Is Not

This is staged inference and dataset generation, not model fine-tuning. It creates a large, resumable table of generated summaries. After that table exists, we can build a reviewed gold set and decide whether fine-tuning is necessary.
