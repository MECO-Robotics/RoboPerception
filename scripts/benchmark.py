#!/usr/bin/env python3
"""Benchmark model on a video; use YOLO26s only if accuracy gain merits latency cost."""
import argparse
import json
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(os.getenv("FRC_DATA_ROOT", "/home/brian/Projects/RoboPerception-data")) / "ultralytics-config"))
os.environ.setdefault("MIOPEN_DEBUG_GCN_ASM_KERNELS", "0")
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
from ultralytics import YOLO
from frcvision.video import process_video

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint")
ap.add_argument("video")
ap.add_argument("--device", default="0")
ap.add_argument("--out", default=str(Path(os.getenv("FRC_DATA_ROOT", "/home/brian/Projects/RoboPerception-data")) / "processed/benchmark.mp4"))
args = ap.parse_args()
result = process_video(YOLO(args.checkpoint), args.video, args.out, device=args.device)
print(json.dumps(result, indent=2))
