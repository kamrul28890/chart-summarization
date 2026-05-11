# Chart Summarization Desktop Pipeline

Desktop CUDA runner for the original Colab notebook:

- `Uthpol_DataSet_Pipeline_Qwen2_5_VL_7B_Instruct.ipynb`

The pipeline summarizes chart images with `Qwen/Qwen2.5-VL-7B-Instruct`, optionally translates the English summary to Bangla with `facebook/nllb-200-distilled-600M`, and writes CSV/XLSX outputs.

For this desktop setup, use:

- `run_pipeline.py` for the runnable script
- `README_DESKTOP_RUN.md` for CUDA environment setup and smoke-test commands

The included ZIP is a small test dataset for validating the pipeline before running the full dataset.
