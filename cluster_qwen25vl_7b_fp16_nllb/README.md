# Cluster Qwen2.5-VL 7B FP16 + NLLB Package

This folder is ready to upload to a Jupyter cluster environment.

## Contents

- `run_qwen25vl_7b_fp16_nllb_cluster.ipynb`: Colab-style notebook for Qwen2.5-VL 7B FP16 chart summarization plus NLLB Bangla translation.
- `dataset_parts/part_01.zip` through `dataset_parts/part_10.zip`: full dataset split into 10 nearly equal ZIP shards.
- `manifests/`: per-part manifests and a full manifest for auditing.
- `merge_cluster_outputs.py`: merges part CSV outputs into one final CSV/XLSX.

The GitHub repository version intentionally excludes the ZIP shards because each shard is larger than GitHub's normal 100 MB file limit. The local folder on this machine still contains the ZIP files for direct cluster upload.

## Dataset split

Total images: 84,363

- `part_01.zip`: 8,437 images
- `part_02.zip`: 8,437 images
- `part_03.zip`: 8,437 images
- `part_04.zip`: 8,436 images
- `part_05.zip`: 8,436 images
- `part_06.zip`: 8,436 images
- `part_07.zip`: 8,436 images
- `part_08.zip`: 8,436 images
- `part_09.zip`: 8,436 images
- `part_10.zip`: 8,436 images

## How to run

1. Upload this whole folder to the cluster.
2. Open `run_qwen25vl_7b_fp16_nllb_cluster.ipynb` from this folder.
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
