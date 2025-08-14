from typing import Any, Dict, Optional
from pathlib import Path
from .base import ModelAdapter
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MobileNetAdapter(ModelAdapter):
	ALLOWED = {"mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large"}

	def __init__(self, model_name: str):
		if model_name not in self.ALLOWED:
			raise ValueError(f"Unsupported MobileNet variant: {model_name}")
		self.model_name = model_name
		try:
			import torch  # type: ignore
			from torchvision import models as tv_models, datasets as tv_datasets, transforms as tv_transforms  # type: ignore
		except Exception as e:
			raise RuntimeError("PyTorch and torchvision are required for MobileNet. Install with `pip install torch torchvision`. ") from e
		self.torch = torch
		self.tv_models = tv_models
		self.tv_datasets = tv_datasets
		self.tv_transforms = tv_transforms

	def name(self) -> str:
		return self.model_name

	def _create_model(self, num_classes: int):
		ctor = getattr(self.tv_models, self.model_name)
		model = ctor(weights=None)
		import torch.nn as nn  # lazy import ok as torch is present
		if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
			in_features = model.classifier[-1].in_features
			model.classifier[-1] = nn.Linear(in_features, num_classes)
		elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
			in_features = model.classifier.in_features
			model.classifier = nn.Linear(in_features, num_classes)
		else:
			raise RuntimeError("Unsupported MobileNet classifier structure")
		return model

	def train(self, *, data: str, epochs: int, img_size: int, batch_size: int, device: Optional[str], project_dir: str, run_name: Optional[str], extra: Dict[str, Any]) -> Dict[str, Any]:
		train_dir = Path(data) / "train"
		val_dir = Path(data) / "val"
		if not train_dir.exists() or not val_dir.exists():
			raise FileNotFoundError("Expected ImageFolder structure with train/ and val/ directories")

		torch = self.torch
		device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
		device_obj = torch.device(device_str)

		transform = self.tv_transforms.Compose([
			self.tv_transforms.Resize((img_size, img_size)),
			self.tv_transforms.ToTensor(),
			self.tv_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
		])
		train_ds = self.tv_datasets.ImageFolder(str(train_dir), transform=transform)
		val_ds = self.tv_datasets.ImageFolder(str(val_dir), transform=transform)
		from torch.utils.data import DataLoader
		train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
		val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

		model = self._create_model(num_classes=len(train_ds.classes)).to(device_obj)
		import torch.nn as nn
		criterion = nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(model.parameters(), lr=float(extra.get("lr", 1e-3)))

		best_acc = 0.0
		best_path = Path(project_dir) / (run_name or self.model_name)
		best_path.mkdir(parents=True, exist_ok=True)
		best_weights = best_path / "best.pt"

		for epoch in range(epochs):
			model.train()
			total_loss = 0.0
			for images, targets in train_loader:
				images, targets = images.to(device_obj), targets.to(device_obj)
				optimizer.zero_grad()
				outputs = model(images)
				loss = criterion(outputs, targets)
				loss.backward()
				optimizer.step()
				total_loss += loss.item() * images.size(0)

			model.eval()
			correct = 0
			total = 0
			with torch.no_grad():
				for images, targets in val_loader:
					images, targets = images.to(device_obj), targets.to(device_obj)
					outputs = model(images)
					preds = outputs.argmax(dim=1)
					correct += (preds == targets).sum().item()
					total += targets.size(0)
			acc = correct / max(1, total)
			logger.info(f"Epoch {epoch+1}/{epochs} - loss={total_loss/max(1, len(train_ds)):.4f} acc={acc:.4f}")
			if acc > best_acc:
				best_acc = acc
				torch.save({"model_state": model.state_dict(), "classes": train_ds.classes}, best_weights)

		return {
			"run_dir": str(best_path),
			"best_weights": str(best_weights),
			"metrics": {"acc": best_acc},
		}

	def export(self, *, weights_path: str, formats: list[str], device: Optional[str], extra: Dict[str, Any]) -> Dict[str, Any]:
		torch = self.torch
		checkpoint = torch.load(weights_path, map_location="cpu")
		classes = checkpoint.get("classes")
		if classes is None:
			raise RuntimeError("Checkpoint missing class metadata")
		model = self._create_model(num_classes=len(classes))
		model.load_state_dict(checkpoint["model_state"]) 
		model.eval()

		dummy = torch.randn(1, 3, int(extra.get("img_size", 224)), int(extra.get("img_size", 224)))
		artifacts: Dict[str, str] = {}
		for fmt in formats:
			if fmt.lower() == "onnx":
				out = Path(weights_path).with_suffix(".onnx")
				try:
					torch.onnx.export(model, dummy, str(out), input_names=["input"], output_names=["logits"], opset_version=int(extra.get("opset", 13)))
					artifacts["onnx"] = str(out)
				except Exception as e:
					logger.warning(f"ONNX export failed: {e}")
			else:
				logger.warning(f"Export format {fmt} is not supported natively by MobileNet adapter. Try ONNX.")
		return artifacts

	@classmethod
	def supports(cls, model_name: str) -> bool:
		return model_name in cls.ALLOWED