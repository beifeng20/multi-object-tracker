"""Matching detections to tracks: IoU cost + Hungarian algorithm."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise intersection-over-union of boxes a (N,4) and b (M,4) in x1,y1,x2,y2 format."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = a[:, None, :4], b[None, :, :4]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def match(dets: np.ndarray, preds: np.ndarray, iou_threshold: float = 0.3):
    """Globally optimal one-to-one assignment maximizing total IoU (Hungarian algorithm, O(n^3)).

    Returns (matches as (det_idx, track_idx) pairs, unmatched detection indices, unmatched track indices).
    Pairs below the IoU threshold are rejected even if the assignment chose them.
    """
    if len(dets) == 0 or len(preds) == 0:
        return [], list(range(len(dets))), list(range(len(preds)))
    iou = iou_matrix(dets, preds)
    rows, cols = linear_sum_assignment(-iou)
    matches = [(r, c) for r, c in zip(rows, cols) if iou[r, c] >= iou_threshold]
    md, mt = {r for r, _ in matches}, {c for _, c in matches}
    return matches, [d for d in range(len(dets)) if d not in md], [t for t in range(len(preds)) if t not in mt]
