"""Bounded OCR sampling from alliance-coloured FRC bumper regions."""
from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import time

import cv2
import numpy as np
from .roster import match_team_number, validate_alliance_teams


class RapidOCRDigits:
    """Read digits from a cropped bumper using the lightweight RapidOCR ONNX runtime."""

    def __init__(self, intra_op_threads=1):
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise RuntimeError(
                "Team-number OCR requires rapidocr-onnxruntime. Install project requirements."
            ) from exc
        self._engine = RapidOCR(
            intra_op_num_threads=int(intra_op_threads),
            inter_op_num_threads=1,
        )

    def read(self, crop):
        return self._read(crop)

    def read_upscaled(self, crop):
        """Retry a weak/rejected crop at larger scale, bounded for CPU cost."""
        if crop is None or crop.size == 0:
            return "", 0.0
        height, width = crop.shape[:2]
        scale = min(3.0, 1600.0 / max(height, width))
        if scale > 1.05:
            crop = cv2.resize(
                crop,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_CUBIC,
            )
        return self._read(crop)

    def _read(self, crop):
        if crop is None or crop.size == 0:
            return "", 0.0
        result, _ = self._engine(crop, use_cls=False)
        if not result:
            return "", 0.0
        candidates = []
        for polygon, text, score in result:
            digits = re.sub(r"\D", "", str(text))
            if not digits or len(digits) > 5:
                continue
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            center_y = float(points[:, 1].mean())
            left_x = float(points[:, 0].min())
            candidates.append((center_y, left_x, digits, float(score)))
        if not candidates:
            return "", 0.0
        candidates.sort(key=lambda item: (item[0], item[1]))
        # A perspective-tilted number can be returned as adjacent text boxes.
        # Join only boxes on the same text line and keep the combined result bounded.
        lines = []
        for candidate in candidates:
            if not lines or abs(candidate[0] - lines[-1][0][0]) > max(8, crop.shape[0] * .22):
                lines.append([candidate])
            else:
                lines[-1].append(candidate)
        line = max(lines, key=lambda items: sum(item[3] for item in items))
        number = "".join(item[2] for item in line)
        if len(number) > 5:
            best = max(candidates, key=lambda item: item[3])
            return best[2], best[3]
        return number, sum(item[3] for item in line) / len(line)


def alliance_bumper_crop(frame, box):
    """Find a horizontal red/blue bumper panel in the lower robot box.

    Restricting OCR to a physically plausible alliance-coloured panel prevents
    text on shirts, signs, and field equipment from becoming team-number reads.
    Returns ``(crop, alliance)`` or ``(None, None)`` when no panel is visible.
    """
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in box)
    box_width, box_height = x2 - x1, y2 - y1
    if box_width < 30 or box_height < 40:
        return None, None
    left = max(0, int(x1))
    right = min(frame_width, int(x2))
    top = max(0, int(y1 + box_height * .48))
    bottom = min(frame_height, int(y2 + box_height * .02))
    roi = frame[top:bottom, left:right]
    if roi.shape[0] < 12 or roi.shape[1] < 24:
        return None, None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 65, 35]), np.array([14, 255, 255])),
        cv2.inRange(hsv, np.array([166, 65, 35]), np.array([180, 255, 255])),
    )
    blue = cv2.inRange(hsv, np.array([88, 55, 30]), np.array([140, 255, 255]))
    kernel_width = max(5, min(17, roi.shape[1] // 12))
    kernel_height = max(3, min(9, roi.shape[0] // 9))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))

    best = None
    for alliance, mask in (("red", red), ("blue", blue)):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            cx, cy, width, height = cv2.boundingRect(contour)
            if area < max(55, roi.shape[0] * roi.shape[1] * .025):
                continue
            if width < max(20, int(roi.shape[1] * .24)) or height < 7:
                continue
            aspect = width / max(1, height)
            if aspect < 1.15:
                continue
            score = area * min(aspect, 4.0)
            if best is None or score > best[0]:
                best = (score, alliance, cx, cy, width, height)
    if best is None:
        return None, None

    _, alliance, cx, cy, width, height = best
    pad_x = max(3, int(width * .035))
    pad_y = max(2, int(height * .05))
    crop = roi[max(0, cy - pad_y):min(roi.shape[0], cy + height + pad_y),
               max(0, cx - pad_x):min(roi.shape[1], cx + width + pad_x)]
    return (crop.copy(), alliance) if crop.size else (None, None)


class AsyncTeamNumberReader:
    """One-worker OCR scheduler with saved crops and per-track confirmation."""

    def __init__(self, interval_frames=30, backend=None, evidence_dir=None,
                 min_detection_confidence=.65, alliance_teams=None):
        if interval_frames < 1:
            raise ValueError("OCR interval must be at least one frame")
        self.interval_frames = int(interval_frames)
        self.backend = backend or RapidOCRDigits()
        self.evidence_dir = Path(evidence_dir) if evidence_dir else None
        self.min_detection_confidence = float(min_detection_confidence)
        self.alliance_teams = (
            validate_alliance_teams(alliance_teams) if alliance_teams else None
        )
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="frc-team-ocr")
        self._future = None
        self._last_submitted = {}
        self._last_track_id = -1
        self._votes = defaultdict(lambda: deque(maxlen=8))
        self._stable = {}
        self.events = []
        self.attempts = 0
        self.total_ms = 0.0

    @staticmethod
    def _fallback_crop(frame, box):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (float(value) for value in box)
        box_width, box_height = x2 - x1, y2 - y1
        left = max(0, int(x1 + box_width * .04))
        right = min(width, int(x2 - box_width * .04))
        top = max(0, int(y1 + box_height * .48))
        bottom = min(height, int(y2 + box_height * .02))
        if right - left < 12 or bottom - top < 8:
            return None
        return frame[top:bottom, left:right].copy()

    def _save_crop(self, crop, frame_idx, track_id, attempt_id):
        if self.evidence_dir is None or crop is None:
            return None
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        name = f"attempt-{attempt_id:04d}-frame-{frame_idx:06d}-track-{track_id}.jpg"
        path = self.evidence_dir / name
        if cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 88]):
            return f"{self.evidence_dir.name}/{name}"
        return None

    def _record(self, frame_idx, track, crop, alliance, status, number="",
                confidence=0.0, frame_ms=0.0, attempted=False, error=None):
        attempt_id = len(self.events) + 1
        crop_file = self._save_crop(crop, frame_idx, int(track["track_id"]), attempt_id)
        event = {
            "attempt": attempt_id,
            "frame": int(frame_idx),
            "track_id": int(track["track_id"]),
            "number": str(number),
            "confidence": round(float(confidence), 3),
            "status": status,
            "confirmed": False,
            "frame_ms": round(float(frame_ms), 2),
            "alliance": alliance or "unknown",
            "crop_file": crop_file,
        }
        if error:
            event["error"] = error
        self.events.append(event)
        if attempted:
            self.attempts += 1
            self.total_ms += float(frame_ms)
        return event

    def poll(self):
        """Collect completed OCR work without blocking the detector loop."""
        if self._future is None or not self._future.done():
            return
        future, self._future = self._future, None
        try:
            frame_idx, track, crop, alliance, number, confidence, elapsed_ms, error = future.result()
        except Exception as exc:
            self.last_error = str(exc)
            return
        if error:
            self._record(
                frame_idx, track, crop, alliance, "error",
                frame_ms=elapsed_ms, attempted=True, error=error,
            )
            return
        raw_number = str(number or "")
        if not raw_number or confidence < .65:
            event = self._record(
                frame_idx, track, crop, alliance, "unreadable",
                number=raw_number, confidence=confidence, frame_ms=elapsed_ms,
                attempted=True,
            )
            event["raw_number"] = raw_number
            return

        matched_number = raw_number
        match_type = "unvalidated"
        if self.alliance_teams:
            matched_number, match_type = match_team_number(
                raw_number, alliance, self.alliance_teams,
            )
            if matched_number is None:
                event = self._record(
                    frame_idx, track, crop, alliance, match_type,
                    number=raw_number, confidence=confidence, frame_ms=elapsed_ms,
                    attempted=True,
                )
                event["raw_number"] = raw_number
                event["match_type"] = match_type
                return

        if not self.alliance_teams and len(raw_number) < 2:
            event = self._record(
                frame_idx, track, crop, alliance, "unreadable",
                number=raw_number, confidence=confidence, frame_ms=elapsed_ms,
                attempted=True,
            )
            event["raw_number"] = raw_number
            return
        event = self._record(
            frame_idx, track, crop, alliance,
            "candidate_partial" if match_type == "partial" else "candidate",
            number=matched_number, confidence=confidence, frame_ms=elapsed_ms,
            attempted=True,
        )
        event["raw_number"] = raw_number
        event["match_type"] = match_type

        track_id = int(track["track_id"])
        self._votes[track_id].append((matched_number, confidence))
        counts = defaultdict(list)
        for value, score in self._votes[track_id]:
            counts[value].append(score)
        candidate, scores = max(counts.items(), key=lambda item: (len(item[1]), sum(item[1])))
        if len(scores) >= 2 and sum(scores) / len(scores) >= .72:
            self._stable[track_id] = (candidate, sum(scores) / len(scores))
            event["confirmed"] = True
            event["status"] = "confirmed"

    def schedule(self, frame_idx, frame, tracks, region_by_track=None):
        """Schedule one best-visible bumper crop; never grows a worker backlog."""
        if self._future is not None:
            return
        eligible = []
        for track in tracks:
            track_id = int(track["track_id"])
            box = track["box"]
            width = float(box[2]) - float(box[0])
            height = float(box[3]) - float(box[1])
            last = self._last_submitted.get(track_id, -self.interval_frames)
            if (track["confidence"] >= self.min_detection_confidence
                    and width >= 34 and height >= 42
                    and frame_idx - last >= self.interval_frames):
                eligible.append(track)
        if not eligible:
            return
        eligible.sort(key=lambda item: int(item["track_id"]))
        chosen = next(
            (item for item in eligible if int(item["track_id"]) > self._last_track_id),
            eligible[0],
        )
        track_id = int(chosen["track_id"])
        self._last_submitted[track_id] = int(frame_idx)
        self._last_track_id = track_id

        color_crop, alliance = alliance_bumper_crop(frame, chosen["box"])
        if color_crop is None:
            evidence_crop = self._fallback_crop(frame, chosen["box"])
            self._record(frame_idx, chosen, evidence_crop, None, "no_bumper_roi")
            return
        self._future = self._executor.submit(
            self._read, int(frame_idx), dict(chosen), color_crop, alliance,
        )

    def _read(self, frame_idx, track, crop, alliance):
        started = time.perf_counter()
        error = None
        try:
            number, confidence = self.backend.read(crop)
            if hasattr(self.backend, "read_upscaled"):
                retry = not number or confidence < .65
                if self.alliance_teams and number:
                    match, _ = match_team_number(
                        number, alliance, self.alliance_teams,
                    )
                    retry = retry or match is None
                if retry:
                    try:
                        retry_number, retry_confidence = self.backend.read_upscaled(crop)
                    except Exception:
                        retry_number, retry_confidence = "", 0.0
                    retry_matches = bool(retry_number) and retry_confidence >= .65
                    if retry_matches and self.alliance_teams:
                        retry_match, _ = match_team_number(
                            retry_number, alliance, self.alliance_teams,
                        )
                        retry_matches = retry_match is not None
                    elif retry_matches:
                        retry_matches = len(str(retry_number)) >= 2
                    if retry_matches:
                        number, confidence = retry_number, retry_confidence
        except Exception as exc:
            number, confidence = "", 0.0
            error = str(exc)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return frame_idx, track, crop, alliance, number, confidence, elapsed_ms, error

    def number_for(self, track_id):
        return self._stable.get(int(track_id), ("", 0.0))

    def close(self):
        if self._future is not None:
            try:
                self._future.result()
                self.poll()
            except Exception as exc:
                self.last_error = str(exc)
        self._executor.shutdown(wait=True)
