# Datasets

This folder contains the small publishable sample dataset used for the completed 200-image run.

## Included

```text
datasets/
`-- sample/
    `-- antu_todo_200_charts.zip
```

The sample ZIP is small enough to keep in GitHub and is the default input for `run_pipeline.py`.

## Full Local Dataset

The full dataset has already been placed locally by the project owner as:

```text
chart images/
chart images.zip
```

That full copy is intentionally excluded from Git because it is approximately 5.3 GB and contains more than 84,000 image files. A normal GitHub repository should not store that directly. Future public releases should use an external dataset host or Git LFS with confirmed quota.

To run against the full local folder:

```powershell
python run_pipeline.py --input "chart images" --qwen-max-pixels 401408
```
