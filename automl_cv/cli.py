import argparse
from typing import Optional, List
from pathlib import Path
from .config import load_config, TrainingConfig
from .trainers.automl_trainer import train_automl, export_model
from .utils.logging import get_logger

logger = get_logger(__name__)


def _cmd_quickstart(args: argparse.Namespace) -> None:
	model_list = [m.strip() for m in args.models.split(",") if m.strip()]
	export_list = [e.strip() for e in args.export.split(",") if e.strip()]
	results = train_automl(
		task=args.task,
		data=args.data,
		models=model_list,
		epochs=args.epochs,
		img_size=args.img_size,
		batch_size=args.batch_size,
		device=args.device,
		project_dir=args.project_dir,
		name=args.name,
		extra={},
	)
	if export_list:
		for r in results:
			logger.info(f"Exporting {r.model_name} best weights: {r.best_weights}")
			artifacts = export_model(r.model_name, r.best_weights, export_list, args.device, {})
			for fmt, path in artifacts.items():
				logger.info(f"Exported {fmt}: {path}")


def _cmd_train(args: argparse.Namespace) -> None:
	cfg: TrainingConfig = load_config(args.config)
	results = train_automl(
		task=cfg.task,
		data=cfg.data,
		models=cfg.models,
		epochs=cfg.epochs,
		img_size=cfg.img_size,
		batch_size=cfg.batch_size,
		device=cfg.device,
		project_dir=cfg.project_dir,
		name=cfg.name,
		extra=cfg.extra,
	)
	if cfg.export:
		for r in results:
			logger.info(f"Exporting {r.model_name} best weights: {r.best_weights}")
			artifacts = export_model(r.model_name, r.best_weights, cfg.export, cfg.device, cfg.extra)
			for fmt, path in artifacts.items():
				logger.info(f"Exported {fmt}: {path}")


def _cmd_export(args: argparse.Namespace) -> None:
	weights = args.weights
	if not weights:
		if not args.run_dir:
			raise SystemExit("Provide either --weights or --run-dir")
		weights = str(Path(args.run_dir) / "weights" / "best.pt")
	format_list = [f.strip() for f in args.formats.split(",") if f.strip()]
	artifacts = export_model(args.model, weights, format_list, args.device, {})
	for fmt, path in artifacts.items():
		logger.info(f"Exported {fmt}: {path}")


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(prog="automl_cv", description="AutoML-CV CLI")
	subparsers = parser.add_subparsers(dest="command", required=True)

	p_qs = subparsers.add_parser("quickstart", help="Quickstart training")
	p_qs.add_argument("--task", required=True, choices=["detection", "classification"], help="Task type")
	p_qs.add_argument("--data", required=True, help="Path to data.yaml (detection) or dataset root (classification)")
	p_qs.add_argument("--models", required=True, help="Comma-separated model names e.g., yolo11n,yolov8n,yolov5n or mobilenet_v3_small")
	p_qs.add_argument("--epochs", type=int, default=20, help="Epochs")
	p_qs.add_argument("--img-size", type=int, default=640, dest="img_size", help="Image size")
	p_qs.add_argument("--batch-size", type=int, default=16, dest="batch_size", help="Batch size")
	p_qs.add_argument("--device", default=None, help="Device index or string (e.g., 'cpu', '0')")
	p_qs.add_argument("--export", default="", help="Comma-separated export formats, e.g., onnx,engine")
	p_qs.add_argument("--project-dir", default="runs", dest="project_dir", help="Project dir for runs")
	p_qs.add_argument("--name", default=None, help="Run name")
	p_qs.set_defaults(func=_cmd_quickstart)

	p_train = subparsers.add_parser("train", help="Train from YAML config")
	p_train.add_argument("--config", required=True, help="Path to YAML config")
	p_train.set_defaults(func=_cmd_train)

	p_exp = subparsers.add_parser("export", help="Export a trained model")
	p_exp.add_argument("--weights", default=None, help="Path to weights file. If omitted, use --run-dir")
	p_exp.add_argument("--run-dir", default=None, help="Run directory containing weights/best.pt")
	p_exp.add_argument("--model", required=True, help="Model name (e.g., yolo11n, yolov5n, mobilenet_v3_small)")
	p_exp.add_argument("--formats", required=True, help="Comma-separated formats, e.g., onnx,engine")
	p_exp.add_argument("--device", default=None, help="Device index or 'cpu'")
	p_exp.set_defaults(func=_cmd_export)

	return parser


def main(argv: Optional[List[str]] = None) -> None:
	parser = build_parser()
	args = parser.parse_args(argv)
	args.func(args)