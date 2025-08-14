from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import yaml


@dataclass
class TrainingConfig:
	task: str  # "detection" or "classification"
	data: str
	models: List[str]
	epochs: int = 20
	img_size: int = 640
	batch_size: int = 16
	device: Optional[str] = None
	export: List[str] = field(default_factory=list)
	project_dir: str = "runs"
	name: Optional[str] = None
	extra: Dict[str, Any] = field(default_factory=dict)


def load_config(path: str) -> TrainingConfig:
	with open(path, "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)
	return TrainingConfig(
		task=cfg["task"],
		data=cfg["data"],
		models=cfg.get("models", []),
		epochs=cfg.get("epochs", 20),
		img_size=cfg.get("img_size", 640),
		batch_size=cfg.get("batch_size", 16),
		device=cfg.get("device"),
		export=cfg.get("export", []),
		project_dir=cfg.get("project_dir", "runs"),
		name=cfg.get("name"),
		extra=cfg.get("extra", {}),
	)