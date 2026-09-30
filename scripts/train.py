#!/usr/bin/env python3
"""GPU-only YOLO26 fine tuning. Runs a kernel preflight before loading data/model."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config/train.yaml"))
    ap.add_argument("--model")
    ap.add_argument("--device", type=int)
    ap.add_argument("--name")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    data_root = Path(os.getenv("FRC_DATA_ROOT", "/home/brian/Projects/RoboPerception-data")).resolve()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(data_root / "ultralytics-config"))
    default_root = "/home/brian/Projects/RoboPerception-data"
    for key in ("data", "project"):
        cfg[key] = str(cfg[key]).replace(default_root, str(data_root))
    cfg.update({k: v for k, v in {"model": args.model, "device": args.device, "name": args.name}.items() if v is not None})
    # The host's gfx900 card lacks the legacy MIOpen grouped-convolution library.
    # The validated non-assembly solver path supports forward and backward kernels.
    os.environ.setdefault("MIOPEN_DEBUG_GCN_ASM_KERNELS", "0")
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
    preflight_env = os.environ.copy()
    preflight_env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + preflight_env.get("PYTHONPATH", "")
    preflight = subprocess.run([sys.executable, "-m", "frcvision.device", "--device", str(cfg["device"])], env=preflight_env)
    if preflight.returncode:
        raise SystemExit("ROCm GPU preflight failed; training did not start")
    from ultralytics import YOLO
    model = YOLO(cfg["model"])
    model.train(data=cfg["data"], project=cfg["project"], name=cfg["name"], imgsz=cfg["imgsz"], epochs=cfg["epochs"], batch=cfg["batch"], workers=cfg["workers"], device=cfg["device"], seed=cfg["seed"], patience=cfg["patience"], amp=cfg["amp"], exist_ok=cfg["exist_ok"])


if __name__ == "__main__":
    main()
