#!/usr/bin/env python3
"""Fetch public FRC detection data and normalize robot classes to frc_robot."""
import argparse
import json
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import requests
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/datasets.yaml")
    ap.add_argument("--root", help="Dataset root (default: $FRC_DATA_ROOT/datasets)")
    ap.add_argument("--source", default="frc-robot-detection-github")
    ap.add_argument("--api-key", default=os.getenv("ROBOFLOW_API_KEY"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-fraction", type=float, default=.2)
    args = ap.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    source = next((s for s in config["sources"] if s["name"] == args.source), None)
    if not source:
        ap.error(f"Unknown source: {args.source}")
    if source.get("type", "roboflow") == "roboflow" and not args.api_key:
        ap.error("Set ROBOFLOW_API_KEY (free Roboflow account key) or pass --api-key")
    root = Path(args.root or (Path(os.getenv("FRC_DATA_ROOT", "/home/brian/Projects/RoboPerception-data")) / "datasets")).resolve()
    raw = root / "raw" / source["name"]
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "dataset.zip"
    if source.get("type") == "github_sparse":
        repo_dir = raw / "repository"
        expected = repo_dir / source["subdirectory"] / "data.yaml"
        if not expected.exists():
            repo_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                "--branch", source.get("ref", "main"), source["repository"], str(repo_dir),
            ], check=True)
            if source.get("revision"):
                subprocess.run(["git", "-C", str(repo_dir), "checkout", "--detach", source["revision"]], check=True)
            subprocess.run(["git", "-C", str(repo_dir), "sparse-checkout", "set", source["subdirectory"]], check=True)
        source_root = repo_dir / source["subdirectory"]
    else:
        source_root = raw
        url = f"https://api.roboflow.com/{source['workspace']}/{source['project']}/{source['version']}/{source['format']}"
    if source.get("type", "roboflow") == "roboflow" and not (raw / "data.yaml").exists():
        params = {"api_key": args.api_key, "nocache": "true"}
        deadline = time.monotonic() + 900
        while True:
            resp = requests.get(url, params=params, timeout=120)
            if resp.status_code == 202:
                if time.monotonic() >= deadline:
                    raise SystemExit("Roboflow export did not finish within 15 minutes")
                time.sleep(5)
                continue
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("ready") is False:
                if time.monotonic() >= deadline:
                    raise SystemExit("Roboflow export did not finish within 15 minutes")
                time.sleep(5)
                continue
            link = payload.get("export", {}).get("link")
            if not link:
                raise SystemExit(f"Roboflow export response had no download link: {payload}")
            break
        with requests.get(link, stream=True, timeout=120) as download:
            download.raise_for_status()
            with archive.open("wb") as f:
                for chunk in download.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(raw)
    data_file = next(source_root.rglob("data.yaml"), None)
    if data_file is None:
        raise SystemExit(f"No data.yaml in Roboflow export at {raw}")
    metadata = yaml.safe_load(data_file.read_text())
    names = metadata.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    # Only labels that denote a robot are retained; misc objects are discarded.
    robot_ids = {i for i, n in enumerate(names) if "robot" in str(n).lower() or str(n).lower() in {"red-bot", "blue-bot", "bot"}}
    if not robot_ids:
        raise SystemExit(f"No robot-named classes in source labels: {names}")
    out = root / "frc_robot"
    if out.exists():
        shutil.rmtree(out)
    image_files = []
    for p in source_root.rglob("*.jpg"):
        if any(part in {"train", "valid", "val", "test"} for part in p.parts):
            image_files.append(p)
    for ext in ("*.jpeg", "*.png", "*.webp"):
        image_files += [p for p in source_root.rglob(ext) if any(x in {"train", "valid", "val", "test"} for x in p.parts)]
    # Preserve Roboflow's existing train/validation/test partitions. The deterministic image-level
    # split is only a fallback for exports with no usable holdout partition.
    import random
    partition_for = {image: next((part for part in image.relative_to(source_root).parts if part in {"train", "valid", "val", "test"}), None) for image in image_files}
    has_train = any(partition_for[p] == "train" for p in image_files)
    has_holdout = any(partition_for[p] in {"valid", "val", "test"} for p in image_files)
    has_val = any(partition_for[p] in {"valid", "val"} for p in image_files)
    if has_train and has_val:
        val_set = {p for p in image_files if partition_for[p] in {"valid", "val"}}
        test_set = {p for p in image_files if partition_for[p] == "test"}
        train_files = [p for p in image_files if partition_for[p] == "train"]
    elif has_train and has_holdout:
        val_set = {p for p in image_files if partition_for[p] in {"valid", "val", "test"}}
        test_set = set()
        train_files = [p for p in image_files if partition_for[p] == "train"]
    else:
        rng = random.Random(args.seed)
        rng.shuffle(image_files)
        val_count = max(1, round(len(image_files) * args.val_fraction))
        val_set = set(image_files[:val_count])
        test_set = set()
        train_files = [p for p in image_files if p not in val_set]
    if not train_files or not val_set:
        raise SystemExit("Dataset must produce non-empty train and validation partitions")
    for image in train_files + list(val_set) + list(test_set):
        rel = image.relative_to(source_root)
        rel_parts = list(rel.parts)
        try:
            rel_parts[rel_parts.index("images")] = "labels"
            label = source_root.joinpath(*rel_parts).with_suffix(".txt")
        except ValueError:
            label = None
        if label is None or not label.exists():
            continue
        rows = []
        for line in label.read_text().splitlines():
            values = line.split()
            if len(values) == 5 and int(values[0]) in robot_ids:
                rows.append("0 " + " ".join(values[1:]))
        split = "test" if image in test_set else ("val" if image in val_set else "train")
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, out / "images" / split / image.name)
        (out / "labels" / split / f"{image.stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
    data = {"path": str(out), "train": "images/train", "val": "images/val", "names": {0: "frc_robot"}}
    if test_set:
        data["test"] = "images/test"
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    revision = subprocess.run(["git", "-C", str(raw / "repository"), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() if source.get("type") == "github_sparse" else "Roboflow export"
    (out / "source.json").write_text(json.dumps({
        "name": source["name"], "url": source["url"], "license": source["license"], "revision": revision,
        "source_classes": names, "robot_class_ids": sorted(robot_ids),
        "train_images": len(train_files), "validation_images": len(val_set), "test_images": len(test_set),
    }, indent=2) + "\n")
    print(f"Prepared {out}; source revision={revision}; robot source classes={[(i, names[i]) for i in sorted(robot_ids)]}")


if __name__ == "__main__":
    main()
