# Results

This folder stores publishable outputs from completed local runs.

## Current Published Run

```text
qwen2_5_vl_200_chart_summaries.csv
qwen2_5_vl_200_chart_summaries.xlsx
```

Run configuration:

- Input: `datasets/sample/antu_todo_200_charts.zip`
- Number of images: 200
- Vision-language model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Translation model: `facebook/nllb-200-distilled-600M`
- Quantization: 4-bit NF4
- Device map: CUDA GPU 0
- Image budget: `--qwen-max-pixels 401408`
- Runtime: 166.8 minutes
- Average speed: 49.4 seconds/image

Columns:

- `image_name`: source image filename
- `image_path`: local extracted image path at run time
- `english_summary`: generated English chart summary
- `bangla_summary`: generated Bangla translation

The `outputs/` folder is ignored and used only for local scratch outputs. Final files intended for publication should be copied or promoted into `results/` with descriptive names.
