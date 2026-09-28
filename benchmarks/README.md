# Jetson Orin Nano — Radio-Map Ablation: .pth → ONNX → TensorRT → Benchmark

Three-stage deployment + benchmarking pipeline for the four models you selected
from `ablation_study_complete (1).ipynb`:

> **CNN, WNet, PartialConvMAE, GNN**

Place this whole folder inside your cloned `jetson-containers` directory and run
the scripts from inside a JetPack/PyTorch container launched via
`jetson-containers run ...`.

## Files

| File | Purpose |
|------|---------|
| `models_radiomap.py` | Model architectures extracted verbatim from the notebook so `*_best.pth` state-dicts re-load off-Colab. Imported by all scripts. |
| `pth_to_onnx.py` | Stage 1 — convert `CNN_best.pth`, `WNet_best.pth`, `PartialConvMAE_best.pth`, `GNN_best.pth` → `.onnx`. |
| `onnx_to_engine.py` | Stage 2 — `trtexec` builds + profiles each `.onnx` into a fused, autotuned, Orin-Nano-specific `.engine`. |
| `ablation_inference_benchmark.py` | Stage 3 — sequentially benchmark each model on **unseen** data; metrics + visualizations. |

## Pipeline

```bash
# 1) Put your four checkpoints in ./checkpoints/
#    CNN_best.pth  WNet_best.pth  PartialConvMAE_best.pth  GNN_best.pth

# 2) Convert to ONNX (static shape is best for TensorRT)
python3 pth_to_onnx.py --weights-dir ./checkpoints --out-dir ./onnx --static

# 3) Build + profile TensorRT engines (FP16 recommended on Orin Nano)
python3 onnx_to_engine.py --onnx-dir ./onnx --out-dir ./engines --precision fp16

# 4) Benchmark on the held-out (unseen) test split
python3 ablation_inference_benchmark.py \
    --test-dir /path/to/dataset_256/test \
    --engines-dir ./engines \
    --weights-dir ./checkpoints \
    --out-dir ./benchmark_results \
    --max-samples 500
```

## Notes

- **W-Net** is exported through `WNetExportWrapper`, which returns only the
  refined output `out2` (the training `forward()` returns a tuple). The state
  dict from the notebook loads with zero missing/unexpected keys.
- **GNN** contains a dynamic kNN (`cdist`/`topk`/`gather`). It exports to ONNX
  fine; `onnx_to_engine.py` forces a **static** shape for it so TensorRT builds
  cleanly.
- The benchmark uses **TensorRT** when an engine is present and `pycuda`/`tensorrt`
  are importable; otherwise it **falls back to PyTorch** automatically so you can
  validate accuracy anywhere (`--backend torch` to force it).
- Live telemetry (GPU%, RAM, power) is sampled from `tegrastats` during each
  model's run; on non-Jetson hosts the sampler simply no-ops.
- Quality metrics on unseen data: MSE, RMSE, MAE (normalised), RMSE/MAE in dB,
  SSIM, PSNR, and **accuracy** = fraction of pixels within `--tol-db` dB of GT.
- Run `tegrastats` power numbers under a fixed power mode for reproducibility,
  e.g. `sudo nvpmodel -m 0 && sudo jetson_clocks`.
