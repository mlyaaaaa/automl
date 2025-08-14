# AutoML-CV: Automated Small Model Training System

AutoML-CV is a lightweight system to automate training and deployment of small computer vision models. It simplifies the process of selecting models, training, evaluating, and exporting to multiple deployment formats.

## Features
- **Model families**: YOLOv5–YOLOv11 (detection/segmentation via Ultralytics & repo bridge), MobileNet (classification via Torch)
- **AutoML orchestration**: try multiple models under constraints and pick the best
- **Simple CLI**: minimal commands to train/tune/export
- **Multi-format export**:
  - Detection (Ultralytics): ONNX, TensorRT (.engine), OpenVINO, TorchScript, CoreML, TFLite (when supported)
  - Classification (MobileNet): ONNX out-of-the-box (+ optional OpenVINO/CoreML/TFLite when toolchains available)
- **Pluggable adapters**: add new model families with a small adapter class

## Install

- Python 3.9+
- Optional accelerators: CUDA/TensorRT as available

Minimal CLI dependencies:
```bash
pip install -r requirements.txt
```

Optional extras (install when you need them):
```bash
# YOLO (Ultralytics)
pip install ultralytics

# PyTorch + torchvision (for classification and adapters)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# Or your CUDA-specific wheel index if using GPU

# Export toolchains (install on demand)
pip install onnx onnxruntime
pip install coremltools  # macOS only
pip install openvino-dev
pip install tensorflow-cpu  # for TFLite conversion helpers
```

## Quickstart

### Detection (YOLO datasets)
Data must be in YOLO format with a dataset YAML (e.g., `data.yaml`).
```bash
python -m automl_cv quickstart \
  --task detection \
  --data /path/to/data.yaml \
  --models yolo11n,yolov8n,yolov5n \
  --epochs 20 \
  --export onnx,engine
```

### Classification (ImageFolder layout)
Assumes `dataset/train` and `dataset/val` subfolders with class directories.
```bash
python -m automl_cv quickstart \
  --task classification \
  --data /path/to/dataset \
  --models mobilenet_v3_small,mobilenet_v2 \
  --epochs 10 \
  --export onnx
```

### Config-driven training
Use example configs in `examples/configs`:
```bash
python -m automl_cv train --config examples/configs/detection_minimal.yaml
python -m automl_cv train --config examples/configs/classification_minimal.yaml
```

### Export
Export from a saved run directory or explicit weights:
```bash
python -m automl_cv export --run-dir runs/2025-01-01_12-00-00_best --formats onnx,engine
# Or
python -m automl_cv export --weights path/to/weights.pt --model yolo11n --task detection --formats onnx
```

## Project Structure
```
automl_cv/
  cli.py                 # CLI entrypoints (Typer)
  __main__.py            # `python -m automl_cv`
  config.py              # Config parsing
  utils/                 # Logging and system helpers
  models/
    base.py              # Adapter interface
    yolov5_adapter.py    # YOLOv5 bridge (repo/CLI)
    ultralytics_adapter.py # YOLOv8/YOLOv11 via Ultralytics
    mobilenet_adapter.py # Classification via Torch
  trainers/
    automl_trainer.py    # Orchestrates training/eval/selection
  search/
    hyperparam.py        # Optional Optuna tuner (minimal stub)
  export/
    exporters.py         # Shared export helpers (optional)
examples/
  configs/
    detection_minimal.yaml
    classification_minimal.yaml
```

## Notes
- YOLOv8/YOLOv11 support relies on the `ultralytics` package.
- YOLOv5 support is provided via a repo bridge that can auto-clone the official repo and invoke its training CLI.
- TensorRT exports require `trtexec` on PATH. OpenVINO, CoreML, and TFLite exports require their respective toolchains installed.

## License
Apache-2.0