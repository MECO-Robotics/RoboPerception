"""Small per-track temporal smoother for detector boxes and map positions."""
from __future__ import annotations


class TrackSmoother:
    """EMA smooth boxes and briefly retain each track through missed detections.

    ``alpha=1`` disables smoothing. Held entries reuse the last smoothed box and
    expire after ``persistence_frames``; they are marked as unobserved so the UI
    can fade them instead of presenting them as fresh detections.
    """

    def __init__(self, alpha: float = 0.35, persistence_frames: int = 6,
                 confirmation_frames: int = 3):
        if not 0 < alpha <= 1:
            raise ValueError("Smoothing alpha must be in (0, 1]")
        if persistence_frames < 0:
            raise ValueError("Persistence frames cannot be negative")
        if confirmation_frames < 1:
            raise ValueError("Track confirmation must be at least one frame")
        self.alpha = float(alpha)
        self.persistence_frames = int(persistence_frames)
        self.confirmation_frames = int(confirmation_frames)
        self._states = {}

    def discard(self, track_ids):
        """Forget rejected tracks so no stale held marker can reappear."""
        for track_id in track_ids:
            self._states.pop(int(track_id), None)

    def update(self, frame_idx: int, detections):
        """Return observed and briefly held tracks for one frame.

        Each detection is ``(track_id, xyxy, confidence, class_id)``. Results
        are dictionaries with a smoothed ``box``, observation flag, and age.
        """
        output = []
        seen = set()
        for track_id, box, confidence, class_id in detections:
            track_id = int(track_id)
            box = [float(v) for v in box]
            previous = self._states.get(track_id)
            gap = frame_idx - previous["frame"] if previous is not None else None
            if previous is not None and gap <= max(1, self.persistence_frames + 1):
                alpha = self.alpha
                box = [alpha * value + (1 - alpha) * old
                       for value, old in zip(box, previous["box"])]
                hits = previous["hits"] + 1
                confirmed = previous["confirmed"] or hits >= self.confirmation_frames
            else:
                hits = 1
                confirmed = self.confirmation_frames == 1
            state = {
                "box": box,
                "confidence": float(confidence),
                "class_id": int(class_id),
                "frame": int(frame_idx),
                "hits": hits,
                "confirmed": confirmed,
            }
            self._states[track_id] = state
            seen.add(track_id)
            if confirmed:
                output.append({**state, "track_id": track_id, "observed": True, "age": 0})

        if self.persistence_frames:
            expired = []
            for track_id, state in self._states.items():
                if track_id in seen:
                    continue
                age = frame_idx - state["frame"]
                if state["confirmed"] and 0 < age <= self.persistence_frames:
                    output.append({**state, "track_id": track_id, "observed": False, "age": age})
                elif age > self.persistence_frames:
                    expired.append(track_id)
            for track_id in expired:
                del self._states[track_id]
        else:
            for track_id in list(self._states):
                if track_id not in seen:
                    del self._states[track_id]

        return sorted(output, key=lambda item: item["track_id"])
