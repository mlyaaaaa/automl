from typing import Any, Dict, Optional
from pathlib import Path
from .base import ModelAdapter
from ..utils.logging import get_logger

logger = get_logger(__name__)


class UltralyticsYOLOAdapter(ModelAdapter):
	SUPPORTED_PREFIXES = ("yolov8", "yolo8", "yolov9", "yolo9", "yolov10", "yolo10", "yolov11", "yolo11")

	def __init__(self, model_name: str):
		self.model_name = model_name
		try:
			from ultralytics import YOLO  # lazy import
			self._YOLO = YOLO
		except Exception as e:
			raise RuntimeError("Ultralytics is required for YOLOv8/9/10/11 models. Install with `pip install ultralytics`. ") from e

	def name(self) -> str:
		return self.model_name

	def _resolve_model(self):
		return self._YOLO(self.model_name)

	def train(self, *, data: str, epochs: int, img_size: int, batch_size: int, device: Optional[str], project_dir: str, run_name: Optional[str], extra: Dict[str, Any]) -> Dict[str, Any]:
		model = self._resolve_model()
		logger.info(f"Training {self.model_name} for {epochs} epochs @ {img_size}px")
		results = model.train(
			data=data,
			epochs=epochs,
			imgsz=img_size,
			batch=batch_size,
			device=device if device is not None else 0,
			project=project_dir,
			name=run_name or self.model_name,
			**extra
		)
		# Ultralytics returns an object; extract useful paths
		run_dir = str(Path(results.save_dir))
		best_path = str(Path(run_dir) / "weights" / "best.pt")
		metrics = {
			"map50": getattr(results, "metrics", {}).get("metrics/mAP50-95(B)") if hasattr(results, "metrics") else None
		}
		return {
			"run_dir": run_dir,
			"best_weights": best_path,
			"metrics": metrics,
		}

	def export(self, *, weights_path: str, formats: list[str], device: Optional[str], extra: Dict[str, Any]) -> Dict[str, Any]:
		model = self._resolve_model()
		model = model.load(weights_path)
		artifacts: Dict[str, str] = {}
		for fmt in formats:
			fmt_lower = fmt.lower()
			logger.info(f"Exporting {self.model_name} to {fmt_lower}")
			try:
				result_path = model.export(format=fmt_lower, device=device if device is not None else 0, **extra)
				if result_path is not None:
					artifacts[fmt_lower] = str(result_path)
			except Exception as e:
				logger.warning(f"Export failed for format {fmt_lower}: {e}")
		return artifacts

	@classmethod
	def supports(cls, model_name: str) -> bool:
		name = model_name.lower()
		return name.startswith(cls.SUPPORTED_PREFIXES)