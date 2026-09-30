"""Offline YOLO inference with ByteTrack IDs, temporal smoothing, and video output."""
import csv
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import imageio_ffmpeg
import torch
from ultralytics.utils.plotting import colors

from .tracking import TrackSmoother
from .geometry import field_homography, project_image_point, reefscape_forbidden
from .numbers import AsyncTeamNumberReader
from .roster import load_video_roster


ROBOT_CLASS_NAMES = {"frc_robot", "robot", "red_robot", "blue_robot"}
OCR_REGION_CLASS_NAMES = {"ocr_region", "number_region"}


def _normalized_class_name(names, class_id):
    if isinstance(names, dict):
        name = names.get(int(class_id), names.get(str(int(class_id)), str(class_id)))
    else:
        name = names[int(class_id)] if 0 <= int(class_id) < len(names) else str(class_id)
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def process_video(model, video_path, output_path, confidence=.50, device="0", imgsz=640,
                  smoothing_alpha=.35, persistence_frames=6, confirmation_frames=3,
                  image_corners=None, calibration_size=None, exclude_reef=False,
                  field_length_m=17.548225, field_width_m=8.0518, source_video=None,
                  read_team_numbers=False, ocr_interval_frames=30):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Video reports invalid dimensions")
    output_path = Path(output_path)
    roster_source = source_video or video_path
    match_roster = load_video_roster(roster_source)
    team_number_reader = None
    if read_team_numbers:
        try:
            evidence_dir = output_path.with_name(f"{output_path.stem}-ocr-evidence")
            team_number_reader = AsyncTeamNumberReader(
                interval_frames=ocr_interval_frames,
                evidence_dir=evidence_dir,
                alliance_teams=match_roster.get("alliance_teams") if match_roster else None,
            )
        except Exception:
            cap.release()
            raise
    scale = min(1.0, 960 / width)
    output_width = max(2, round(width * scale / 2) * 2)
    output_height = max(2, round(height * scale / 2) * 2)
    processing_corners = None
    output_calibration = None
    if image_corners is not None and calibration_size is not None:
        ref_width, ref_height = calibration_size
        processing_corners = [
            [float(x) * width / ref_width, float(y) * height / ref_height]
            for x, y in image_corners
        ]
        output_calibration = {
            "image_corners": [
                [int(round(float(x) * output_width / width)),
                 int(round(float(y) * output_height / height))]
                for x, y in processing_corners
            ],
            "image_width": output_width,
            "image_height": output_height,
            "field_length_m": field_length_m,
            "field_width_m": field_width_m,
        }
    reef_transform = None
    if exclude_reef:
        if processing_corners is None:
            cap.release()
            raise ValueError("Reef exclusion requires a saved field calibration")
        reef_transform = field_homography(processing_corners, field_length_m, field_width_m)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(output_path), (output_width, output_height), fps=source_fps, codec="libx264",
        pix_fmt_out="yuv420p", quality=5, macro_block_size=1,
        output_params=["-movflags", "+faststart", "-profile:v", "main"],
    )
    writer.send(None)
    csv_path = output_path.with_suffix(".csv")
    detections_path = output_path.with_suffix(".detections.json")
    rows = []
    detection_rows = []
    smoother = TrackSmoother(
        alpha=smoothing_alpha,
        persistence_frames=persistence_frames,
        confirmation_frames=confirmation_frames,
    )
    frame_idx = 0
    class_counts = Counter()
    model_names = model.names
    processing_started = time.perf_counter()
    try:
        while True:
            frame_started = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            start = time.perf_counter()
            result = model.track(
                frame,
                conf=min(.1, confidence),
                device=device,
                imgsz=imgsz,
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
            )[0]
            if str(device) != "cpu" and torch.cuda.is_available():
                sync_device = int(device) if str(device).isdigit() else device
                torch.cuda.synchronize(sync_device)
            latency_ms = (time.perf_counter() - start) * 1000
            boxes = result.boxes
            raw_boxes = boxes.xyxy.detach().cpu().numpy() if len(boxes) else []
            confidences = boxes.conf.detach().cpu().numpy() if len(boxes) else []
            classes = boxes.cls.detach().cpu().numpy() if len(boxes) else []
            track_ids = boxes.id
            if track_ids is None:
                # Auxiliary boxes can still be rendered and used as OCR regions.
                track_ids = [None] * len(raw_boxes)
            else:
                track_ids = track_ids.int().detach().cpu().numpy()

            robot_candidates = []
            auxiliary_detections = []
            ocr_regions = []
            for i, box in enumerate(raw_boxes):
                if confidences[i] < confidence:
                    continue
                class_id = int(classes[i])
                class_name = _normalized_class_name(model_names, class_id)
                class_counts[class_name] += 1
                track_id = track_ids[i]
                if class_name in ROBOT_CLASS_NAMES and track_id is not None:
                    robot_candidates.append((int(track_id), box, confidences[i], class_id))
                else:
                    auxiliary_detections.append((box, float(confidences[i]), class_name))
                    if class_name in OCR_REGION_CLASS_NAMES:
                        ocr_regions.append((box, float(confidences[i])))

            accepted = []
            rejected_reef_tracks = []
            for track_id, box, score, class_id in robot_candidates:
                if reef_transform is not None:
                    floor_point = ((box[0] + box[2]) / 2, box[3])
                    field_point = project_image_point(floor_point, reef_transform)
                    if reefscape_forbidden(field_point, field_length_m, field_width_m):
                        rejected_reef_tracks.append(track_id)
                        continue
                accepted.append((track_id, box, score, class_id))
            smoother.discard(rejected_reef_tracks)
            frame_tracks = smoother.update(frame_idx, accepted)
            observed_tracks = [track for track in frame_tracks if track["observed"]]
            if team_number_reader:
                team_number_reader.poll()
                regions_by_track = {}
                for track in observed_tracks:
                    x1, y1, x2, y2 = track["box"]
                    lower_y = y1 + .42 * (y2 - y1)
                    matches = []
                    for region_box, region_confidence in ocr_regions:
                        rx1, ry1, rx2, ry2 = region_box
                        center_x, center_y = (rx1 + rx2) / 2, (ry1 + ry2) / 2
                        if x1 <= center_x <= x2 and lower_y <= center_y <= y2 + .08 * (y2 - y1):
                            matches.append((region_confidence, region_box))
                    if matches:
                        regions_by_track[int(track["track_id"])] = max(
                            matches, key=lambda item: item[0],
                        )[1]
                team_number_reader.schedule(frame_idx, frame, observed_tracks, regions_by_track)
            for track in frame_tracks:
                if team_number_reader:
                    number, number_confidence = team_number_reader.number_for(track["track_id"])
                    track["team_number"] = number
                    track["number_confidence"] = number_confidence
                else:
                    track["team_number"] = ""
                    track["number_confidence"] = 0.0
            annotated = frame.copy()
            font_scale = max(.42, min(.52, width / 2400))
            for track in observed_tracks:
                x1, y1, x2, y2 = (int(round(value)) for value in track["box"])
                class_name = _normalized_class_name(model_names, track["class_id"])
                color = (55, 75, 245) if class_name == "red_robot" else (235, 145, 55) if class_name == "blue_robot" else colors(track["track_id"], bgr=True)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                class_name = _normalized_class_name(model_names, track["class_id"])
                track_label = (
                    f" Team {track['team_number']} (T{track['track_id']})"
                    if track["team_number"] else f" T{track['track_id']}"
                )
                label = f"{class_name}{track_label} {track['confidence']:.2f}"
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1,
                )
                label_top = max(0, y1 - label_height - baseline - 6)
                cv2.rectangle(
                    annotated, (x1, label_top), (x1 + label_width + 6, y1), color, -1,
                )
                cv2.putText(
                    annotated, label, (x1 + 3, y1 - baseline - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
                )
            for box, score, class_name in auxiliary_detections:
                x1, y1, x2, y2 = (int(round(value)) for value in box)
                color = (80, 220, 165) if class_name == "game_piece" else (30, 205, 245) if class_name in OCR_REGION_CLASS_NAMES else (190, 190, 190)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                label = f"{class_name} {score:.2f}"
                cv2.putText(annotated, label, (x1 + 2, max(14, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
            if (output_width, output_height) != (width, height):
                annotated = cv2.resize(annotated, (output_width, output_height), interpolation=cv2.INTER_AREA)
            writer.send(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            count = len(observed_tracks)
            processing_ms = (time.perf_counter() - frame_started) * 1000
            rows.append((
                frame_idx, count, round(latency_ms, 3),
                round(1000 / latency_ms, 2) if latency_ms else 0,
                round(processing_ms, 3), round(1000 / processing_ms, 2) if processing_ms else 0,
            ))
            scale_x = output_width / width
            scale_y = output_height / height
            for track in frame_tracks:
                box = track["box"]
                x = ((box[0] + box[2]) / 2) * scale_x
                y = box[3] * scale_y
                detection_rows.append([
                    frame_idx, round(x, 2), round(y, 2), round(track["confidence"], 3),
                    track["track_id"], int(track["observed"]), track["age"],
                    track["team_number"], round(track["number_confidence"], 3),
                ])
            frame_idx += 1
    finally:
        cap.release()
        writer.close()
        if team_number_reader:
            team_number_reader.close()
    if frame_idx == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"No frames could be decoded from {video_path}; check the video codec."
        )
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "detections", "inference_ms", "inference_fps",
                    "processing_ms", "processing_fps"])
        w.writerows(rows)
    if output_calibration:
        output_path.with_suffix(".calibration.json").write_text(
            json.dumps(output_calibration, indent=2)
        )
    latency = sum(r[2] for r in rows) / len(rows) if rows else 0
    processing_latency = sum(r[4] for r in rows) / len(rows) if rows else 0
    processing_seconds = time.perf_counter() - processing_started
    team_numbers = (
        [{"track_id": track_id, "number": value[0], "confidence": round(value[1], 3)}
         for track_id, value in sorted(team_number_reader._stable.items())]
        if team_number_reader else []
    )
    ocr_events = team_number_reader.events if team_number_reader else []
    read_events = [event for event in ocr_events if event.get("number")]
    confirmed_events = [event for event in ocr_events if event.get("confirmed")]
    ocr_mean_ms = (team_number_reader.total_ms / team_number_reader.attempts
                   if team_number_reader and team_number_reader.attempts else 0.0)
    ocr_attempts = team_number_reader.attempts if team_number_reader else 0
    with detections_path.open("w") as f:
        json.dump({
            "fps": source_fps,
            "width": output_width,
            "height": output_height,
            "frames": frame_idx,
            "tracker": "ByteTrack",
            "smoothing_alpha": smoothing_alpha,
            "persistence_frames": persistence_frames,
            "confirmation_frames": confirmation_frames,
            "reef_exclusion": bool(exclude_reef),
            "team_number_ocr": bool(read_team_numbers),
            "ocr_interval_frames": int(ocr_interval_frames) if read_team_numbers else None,
            "match_roster": match_roster,
            "team_numbers": team_numbers,
            "number_reads": read_events,
            "ocr_events": ocr_events,
            "class_counts": dict(class_counts),
            "ocr_attempts": ocr_attempts,
            "ocr_samples": len(ocr_events),
            "ocr_candidates": sum(event.get("status") in {
                "candidate", "candidate_partial", "confirmed",
            } for event in ocr_events),
            "ocr_rejected": sum(str(event.get("status", "")).startswith("rejected_")
                                for event in ocr_events),
            "ocr_partial_matches": sum(event.get("match_type") == "partial"
                                       for event in ocr_events),
            "ocr_skipped_no_bumper": sum(event.get("status") == "no_bumper_roi"
                                          for event in ocr_events),
            "ocr_mean_ms": ocr_mean_ms,
            "source_video": str(Path(source_video).resolve()) if source_video else str(Path(video_path).resolve()),
            "reef_filter_basis": "calibrated robot box bottom-center within Reef hardware footprint" if exclude_reef else None,
            "positions": detection_rows,
        }, f, separators=(",", ":"))
    return {
        "frames": frame_idx,
        "detections": sum(r[1] for r in rows),
        "mean_latency_ms": latency,
        "inference_fps": 1000 / latency if latency else 0,
        "mean_processing_ms": processing_latency,
        "processing_fps": frame_idx / processing_seconds if processing_seconds else 0,
        "source_fps": source_fps,
        "ocr_mean_ms": ocr_mean_ms,
        "ocr_attempts": ocr_attempts,
        "ocr_samples": len(ocr_events),
        "ocr_candidates": sum(event.get("status") in {
            "candidate", "candidate_partial", "confirmed",
        } for event in ocr_events),
        "ocr_rejected": sum(str(event.get("status", "")).startswith("rejected_")
                            for event in ocr_events),
        "ocr_partial_matches": sum(event.get("match_type") == "partial"
                                   for event in ocr_events),
        "ocr_confirmed": len(confirmed_events),
        "team_number_ocr": bool(read_team_numbers),
        "ocr_interval_frames": int(ocr_interval_frames) if read_team_numbers else None,
        "match_roster": match_roster,
        "team_numbers": team_numbers,
        "number_reads": len(read_events),
        "class_counts": dict(class_counts),
        "annotated": str(output_path),
        "metrics": str(csv_path),
        "detections_file": str(detections_path),
        "width": output_width,
        "height": output_height,
        "tracker": "ByteTrack",
        "smoothing_alpha": smoothing_alpha,
        "persistence_frames": persistence_frames,
        "confirmation_frames": confirmation_frames,
        "reef_exclusion": bool(exclude_reef),
    }
