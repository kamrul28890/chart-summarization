# Desktop CUDA Run

This project was converted from the Colab notebook into `run_pipeline.py`.

## Machine Settings Detected

- GPU: NVIDIA GeForce RTX 3070, 8 GB VRAM
- CPU: Intel Core i7-13700KF
- RAM: about 51 GB
- NVIDIA driver: 591.86
- Driver-reported CUDA support: 13.1
- Current global Python: 3.10.11
- Current global PyTorch: CPU-only, so it cannot run CUDA

The notebook did not use quantization. It loaded Qwen with:

```python
torch_dtype=torch.float16,
device_map="auto"
```

For this desktop, the default runner uses 4-bit NF4 bitsandbytes quantization because plain FP16 Qwen2.5-VL-7B is too large for an 8 GB display GPU.

## Setup

Use a clean virtual environment instead of the global Python install:

```powershell
cd "D:\My Projects\chart-summarization-sust"
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Then verify CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected result: `torch.cuda.is_available()` should print `True`.

## Smoke Test

Close Ollama and other GPU-heavy apps first if possible. Then run 2 images:

```powershell
python run_pipeline.py --limit 2 --max-gpu-memory 5GiB
```

## FP16 Test

The notebook used unquantized FP16. To test that path directly on GPU 0:

```powershell
python run_pipeline_fp16.py
```

This defaults to one image, no quantization, and `--device-map cuda`. On an RTX 3070 8 GB, this may fail with CUDA out-of-memory. That is useful information: it means FP16 does not fit as a pure GPU run on this card.

To allow CPU offload while still using FP16 weights:

```powershell
python run_pipeline.py --quantization none --device-map auto --limit 1 --max-gpu-memory 5GiB
```

Outputs are written to:

```text
outputs\partial_summaries.csv
outputs\testset_summaries_1.xlsx
```

## Full Test ZIP

The local ZIP contains 200 images. After the smoke test works:

```powershell
python run_pipeline.py --max-gpu-memory 5GiB
```

If VRAM is mostly free after closing apps, try:

```powershell
python run_pipeline.py --max-gpu-memory 6GiB
```

If the run hits CUDA out-of-memory, lower it:

```powershell
python run_pipeline.py --max-gpu-memory 4GiB
```
