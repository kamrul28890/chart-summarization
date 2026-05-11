# Project Status

## Research Goal

The goal is to build a practical chart-summarization workflow that converts chart images into readable textual explanations. The intended output is useful for dataset preparation, accessibility, chart indexing, and downstream reporting.

## Plain-English Explanation

We give the system a chart image. The model reads the chart visually, identifies the main trend or comparison, and writes a short explanation. A second model can translate that explanation into Bangla. The final spreadsheet gives one row per chart, with the original filename and generated summaries.

## Work Completed

- The original notebook workflow was preserved under `notebooks/`.
- The main workflow was converted into `run_pipeline.py` for repeatable command-line execution.
- The project was tested on a Windows desktop with an NVIDIA RTX 3070 8 GB GPU.
- GPU memory pressure was reduced by stopping background GPU-heavy processes and using 4-bit model quantization.
- A successful 200-image run was completed with Qwen2.5-VL and NLLB translation.
- Final CSV and XLSX results were saved under `results/`.
- The full local dataset was identified as too large for normal GitHub storage and excluded from the repository.

## Current Limitations

- Full-dataset processing has not yet been run end-to-end.
- The current script processes images sequentially and does not skip completed rows automatically.
- The generated summaries have not yet been scored against human references.
- Translation quality has not yet been separately evaluated.
- The current prompt is fixed; prompt variants have not yet been compared.

## Recommended Next Steps

1. Add resume support using `image_name` or image hash so large runs can continue after interruption.
2. Run the full local dataset in manageable batches.
3. Add quality-control checks for empty outputs, repeated boilerplate, unusually short summaries, and translation failures.
4. Create a manually reviewed validation subset.
5. Compare model outputs against reference summaries using both automatic metrics and human review.
6. Add run manifests under `results/` for every published experiment.
7. Consider hosting the full dataset externally and linking it from `datasets/README.md`.

## Publishing Decision

The full dataset is local-only for this GitHub version. This keeps the repository usable and avoids pushing multi-gigabyte data into Git history. The repo includes the code, documentation, sample data, and completed 200-image result artifacts.
