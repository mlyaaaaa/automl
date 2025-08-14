from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class ModelAdapter(ABC):
	@abstractmethod
	def name(self) -> str:
		...

	@abstractmethod
	def train(self, *, data: str, epochs: int, img_size: int, batch_size: int, device: Optional[str], project_dir: str, run_name: Optional[str], extra: Dict[str, Any]) -> Dict[str, Any]:
		"""Train model and return a result dict including metrics and artifact paths."""
		...

	@abstractmethod
	def export(self, *, weights_path: str, formats: list[str], device: Optional[str], extra: Dict[str, Any]) -> Dict[str, Any]:
		"""Export weights to requested formats and return mapping format->path."""
		...