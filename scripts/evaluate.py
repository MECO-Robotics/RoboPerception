#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(os.getenv("FRC_DATA_ROOT", "/home/brian/Projects/RoboPerception-data")) / "ultralytics-config"))
os.environ.setdefault("MIOPEN_DEBUG_GCN_ASM_KERNELS", "0")
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint")
ap.add_argument("--data", default=str(Path(os.getenv("FRC_DATA_ROOT", "/home/brian/Projects/RoboPerception-data")) / "datasets/frc_robot/data.yaml"))
ap.add_argument("--device", default="0")
ap.add_argument("--split", choices=("val", "test"), default="val")
args = ap.parse_args()
data_root = Path(os.getenv("FRC_DATA_ROOT", "/home/brian/Projects/RoboPerception-data"))
metrics = YOLO(args.checkpoint).val(
    data=args.data, device=args.device, split=args.split,
    project=str(data_root / "runs/evaluation"), name=f"{Path(args.checkpoint).stem}_{args.split}", exist_ok=True,
)
print({"precision": float(metrics.box.mp), "recall": float(metrics.box.mr), "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map)})
