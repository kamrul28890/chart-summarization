# Dataset Assessment

Assessment date: 2026-05-11

## Local Dataset

The full local dataset is stored outside Git as:

```text
chart images/
chart images.zip
```

It is intentionally ignored because it is too large for normal GitHub storage.

## Inventory

- Total image files: 84,363
- Total size: 5,300,160,819 bytes, about 5.3 GB
- File type: PNG only
- Directory layout: flat folder, no nested split folders
- Image dimensions: all images are 1500 x 900
- Corrupt/unreadable images: 0 found in header validation

## Filename Families

```text
train_k: 34,702 images, IDs 0-34701, no missing IDs
train_s: 32,786 images, IDs 0-32785, no missing IDs
valid_k: 4,338 images, IDs 0-4337, no missing IDs
valid_s: 4,101 images, IDs 0-4100, no missing IDs
test_k:  4,338 images, IDs 0-4337, no missing IDs
test_s:  4,098 images, IDs 0-4097, no missing IDs
```

The naming suggests there are two chart/data families, `k` and `s`, each with train/validation/test splits.

## Exact Duplicates

SHA-256 duplicate scan found:

- Duplicate hash groups: 39
- Duplicate files involved: 78
- Extra duplicate copies: 39

Duplicate family patterns:

```text
train_k only:              25 groups
train_s only:               5 groups
train_k and valid_k:        4 groups
test_k and train_k:         3 groups
test_k and valid_k:         1 group
train_s and valid_s:        1 group
```

The duplicate rate is very low, but cross-split duplicates should be excluded from evaluation to avoid leakage.

## Existing Processed Sample

The current published result file covers 200 images:

```text
results/qwen2_5_vl_200_chart_summaries.xlsx
results/qwen2_5_vl_200_chart_summaries.csv
```

Quality/completeness signals:

- Rows: 200
- Empty English summaries: 0
- Empty Bangla summaries: 0
- Duplicate English summaries: 0
- Duplicate Bangla summaries: 0
- English summary length: 60-141 words, median 91 words
- Bangla summary length: 50-129 words, median 85 words

The sample is complete and structurally usable, but the summaries have not yet been checked against human references.

## Recommendation

Do not start full model training yet. The next work should be a staged data-and-evaluation workflow:

1. Create a manifest for all 84,363 images with filename, split, family, numeric ID, size, dimensions, and duplicate hash.
2. Remove or mark exact duplicate leakage before any validation/test evaluation.
3. Add resume-aware batch inference so the full dataset can be summarized safely over multiple sessions.
4. Generate summaries for validation/test first, then a larger train subset.
5. Build a small manually reviewed gold set before deciding on fine-tuning.
6. Use Azure/OpenAI APIs for labeling, critique, or refinement if useful, but reserve GPU-cluster training for after the target behavior is clear.

## Why This Order

The dataset is clean enough to process, but training before evaluation would be premature. We first need a reliable manifest, leak-free splits, batch inference, and a small human-reviewed benchmark. Once those exist, we can decide whether prompt refinement, API-based distillation, or actual vision-language-model fine-tuning is the best next step.
