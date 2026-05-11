# Qwen2.5-VL 3B FP16 Comparison

Comparison date: 2026-05-11

## Purpose

The earlier 200-image sample was generated with `Qwen/Qwen2.5-VL-7B-Instruct` using 4-bit NF4 quantization. This comparison reruns the same 200 images with `Qwen/Qwen2.5-VL-3B-Instruct` using FP16 weights and no 4-bit quantization.

## Run Commands

```powershell
python run_pipeline.py `
  --input datasets\sample\antu_todo_200_charts.zip `
  --output-dir outputs\qwen25vl_3b_fp16_sample200 `
  --vl-model-name Qwen/Qwen2.5-VL-3B-Instruct `
  --quantization none `
  --device-map cuda `
  --qwen-max-pixels 401408
```

## Output Files

```text
results/qwen2_5_vl_200_chart_summaries.csv
results/qwen2_5_vl_200_chart_summaries.xlsx
results/qwen2_5_vl_3b_fp16_200_chart_summaries.csv
results/qwen2_5_vl_3b_fp16_200_chart_summaries.xlsx
```

## Runtime

```text
7B 4-bit: 166.8 minutes, 49.4 seconds/image
3B FP16: 242.2 minutes, 72.5 seconds/image
```

On this desktop, 3B FP16 was slower than 7B 4-bit for the 200-image sample.

## Structural Output Comparison

Both runs produced 200 rows in the same image order.

```text
7B 4-bit English summaries:
- empty: 0
- duplicate summaries: 0
- word length: 60-141, median 91

3B FP16 English summaries:
- empty: 0
- duplicate summaries: 0
- word length: 79-158, median 116
```

The 3B FP16 summaries were usually longer:

```text
3B longer than 7B: 182 images
3B shorter than 7B: 16 images
median word delta: +25 words
```

## Initial Interpretation

The 3B FP16 path fits on the desktop GPU and produces complete outputs, but it is not faster in this configuration. Because it is slower and more verbose, the next decision should be based on manual quality review, not model size alone.

Recommended next comparison:

1. Review 25-50 paired examples from both result files.
2. Score factual correctness, trend detection, anomaly handling, and readability.
3. If 3B FP16 is not clearly better, keep 7B 4-bit for full-dataset generation.
4. If 3B FP16 is better but too slow, test a smaller `--max-new-tokens` value or lower image budget before scaling up.
