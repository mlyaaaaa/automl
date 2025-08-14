from typing import Any, Dict, Optional
from pathlib import Path
import subprocess
import sys
import tempfile
from .base import ModelAdapter
from ..utils.logging import get_logger

logger = get_logger(__name__)


class YOLOv5Adapter(ModelAdapter):
	SUPPORTED_PREFIXES = ("yolov5",)

	def __init__(self, model_name: str = "yolov5n"):
		self.model_name = model_name

	def name(self) -> str:
		return self.model_name

	@staticmethod
	def _ensure_repo(repo_dir: Path) -> None:
		if repo_dir.exists():
			return
		repo_dir.parent.mkdir(parents=True, exist_ok=True)
		logger.info("Cloning ultralytics/yolov5 repo for training...")
		subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ultralytics/yolov5.git", str(repo_dir)], check=True)

	def train(self, *, data: str, epochs: int, img_size: int, batch_size: int, device: Optional[str], project_dir: str, run_name: Optional[str], extra: Dict[str, Any]) -> Dict[str, Any]:
		repo_dir = Path(tempfile.gettempdir()) / "automl_cv_repos" / "yolov5"
		self._ensure_repo(repo_dir)
		python_exe = sys.executable
		cmd = [
			python_exe,
			str(repo_dir / "train.py"),
			"--img", str(img_size),
			"--batch", str(batch_size),
			"--epochs", str(epochs),
			"--data", data,
			"--weights", self.model_name,
			"--project", project_dir,
			"--name", run_name or self.model_name,
		]
		if device is not None:
			cmd += ["--device", device]
		for k, v in extra.items():
			cmd += [f"--{k}", str(v)]
		logger.info("Running YOLOv5 training CLI: " + " ".join(cmd))
		subprocess.run(cmd, check=True)
		run_dir = str(Path(project_dir) / (run_name or self.model_name))
		best_path = str(Path(run_dir) / "weights" / "best.pt")
		return {
			"run_dir": run_dir,
			"best_weights": best_path,
			"metrics": {},
		}

	def export(self, *, weights_path: str, formats: list[str], device: Optional[str], extra: Dict[str, Any]) -> Dict[str, Any]:
		# Delegate to Ultralytics YOLOv5 export.py
		repo_dir = Path(tempfile.gettempdir()) / "automl_cv_repos" / "yolov5"
		self._ensure_repo(repo_dir)
		python_exe = sys.executable
		artifacts: Dict[str, str] = {}
		for fmt in formats:
			cmd = [
				python_exe,
				str(repo_dir / "export.py"),
				"--weights", weights_path,
				"--img", "640",
				"--device", device or "0",
				"--include", fmt.lower(),
			]
			for k, v in extra.items():
				cmd += [f"--{k}", str(v)]
			logger.info("YOLOv5 export: " + " ".join(cmd))
			subprocess.run(cmd, check=True)
			# Heuristic: produced file sits alongside weights with extension
			w = Path(weights_path)
			out = str(w.with_suffix(f".{fmt.lower()}"))
			artifacts[fmt.lower()] = out
		return artifacts

	@classmethod
	def supports(cls, model_name: str) -> bool:
		return model_name.lower().startswith(cls.SUPPORTED_PREFIXES)