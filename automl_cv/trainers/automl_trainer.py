from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RunResult:
	model_name: str
	run_dir: str
	best_weights: str
	metrics: Dict[str, Any]


def resolve_adapter(model_name: str):
	name = model_name.lower()
	# YOLOv5
	if name.startswith("yolov5"):
		from ..models.yolov5_adapter import YOLOv5Adapter
		return YOLOv5Adapter(model_name)
	# Ultralytics YOLO (v8/9/10/11)
	if name.startswith(("yolov8", "yolo8", "yolov9", "yolo9", "yolov10", "yolo10", "yolov11", "yolo11")):
		from ..models.ultralytics_adapter import UltralyticsYOLOAdapter
		return UltralyticsYOLOAdapter(model_name)
	# MobileNet classification
	if name.startswith("mobilenet"):
		from ..models.mobilenet_adapter import MobileNetAdapter
		return MobileNetAdapter(model_name)
	raise ValueError(f"No adapter found for model: {model_name}")


def train_automl(*, task: str, data: str, models: List[str], epochs: int, img_size: int, batch_size: int, device: Optional[str], project_dir: str, name: Optional[str], extra: Dict[str, Any]) -> List[RunResult]:
	Path(project_dir).mkdir(parents=True, exist_ok=True)
	results: List[RunResult] = []
	for model_name in models:
		adapter = resolve_adapter(model_name)
		logger.info(f"=== Training candidate: {model_name} ===")
		res = adapter.train(
			data=data,
			epochs=epochs,
			img_size=img_size,
			batch_size=batch_size,
			device=device,
			project_dir=project_dir,
			run_name=name or model_name,
			extra=extra,
		)
		results.append(RunResult(model_name=model_name, run_dir=res["run_dir"], best_weights=res["best_weights"], metrics=res.get("metrics", {})))
	return results


def export_model(model_name: str, weights_path: str, formats: List[str], device: Optional[str], extra: Dict[str, Any]) -> Dict[str, str]:
	adapter = resolve_adapter(model_name)
	return adapter.export(weights_path=weights_path, formats=formats, device=device, extra=extra)