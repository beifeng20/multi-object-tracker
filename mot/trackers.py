"""SORT tracker and a motion-free IoU baseline with the same interface.

update(detections (N,4+) array of x1,y1,x2,y2[,score]) -> list of (track_id, box) for confirmed tracks.
"""
import numpy as np

from .association import match
from .kalman_box import KalmanBox


class _Track:
    def __init__(self, track_id: int, box: np.ndarray, use_kalman: bool):
        self.id = track_id
        self.kf = KalmanBox(box) if use_kalman else None
        self.last_box = np.asarray(box[:4], float)
        self.hits = 1          # number of matched detections
        self.time_since_update = 0

    def predict(self) -> np.ndarray:
        self.time_since_update += 1
        return self.kf.predict() if self.kf else self.last_box

    def update(self, box: np.ndarray) -> None:
        self.time_since_update = 0
        self.hits += 1
        self.last_box = np.asarray(box[:4], float)
        if self.kf:
            self.kf.update(box)

    @property
    def box(self) -> np.ndarray:
        return self.kf.box if self.kf else self.last_box


class Sort:
    """Kalman prediction + Hungarian matching on IoU.

    max_age: frames a track survives without a detection (bridges short occlusions / missed detections)
    min_hits: detections needed before a track is reported (suppresses one-frame false positives)
    coast: also report confirmed tracks in frames without a detection (the predicted box), for up to max_age
           frames — fills gaps from missed detections and short occlusions
    """

    use_kalman = True

    def __init__(self, max_age: int = 5, min_hits: int = 3, iou_threshold: float = 0.3, coast: bool = False):
        self.max_age, self.min_hits, self.iou_threshold, self.coast = max_age, min_hits, iou_threshold, coast
        self.tracks: list[_Track] = []
        self.next_id = 1
        self.frame = 0

    def update(self, dets: np.ndarray) -> list[tuple[int, np.ndarray]]:
        self.frame += 1
        dets = np.asarray(dets, float)
        if dets.size == 0:
            dets = np.zeros((0, 4))
        preds = np.array([t.predict() for t in self.tracks]).reshape(-1, 4)
        matches, new_dets, _ = match(dets, preds, self.iou_threshold)
        for d, t in matches:
            self.tracks[t].update(dets[d])
        for d in new_dets:
            self.tracks.append(_Track(self.next_id, dets[d], self.use_kalman))
            self.next_id += 1
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        out = []
        for t in self.tracks:
            confirmed = t.hits >= self.min_hits or self.frame <= self.min_hits
            if confirmed and (t.time_since_update == 0 or (self.coast and t.hits >= self.min_hits)):
                out.append((t.id, t.box))
        return out


class IouTracker(Sort):
    """Baseline: same matching and track management, but no motion model —
    a track is compared using its last observed box."""

    use_kalman = False
