import numpy as np

from mot.association import iou_matrix, match
from mot.kalman_box import box_to_z, x_to_box
from mot.trackers import IouTracker, Sort


def test_iou_values():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    assert np.allclose(iou_matrix(a, b), [[1.0, 50 / 150, 0.0]])


def test_hungarian_beats_greedy():
    # greedy would give det 0 to track 0 (IoU 0.6) and leave det 1 with nothing good;
    # the optimal assignment pairs det0-track1 and det1-track0
    dets = np.array([[0, 0, 10, 10], [4, 0, 14, 10]], float)
    tracks = np.array([[2, 0, 12, 10], [-3, 0, 7, 10]], float)
    matches, _, _ = match(dets, tracks, iou_threshold=0.3)
    assert sorted(matches) == [(0, 1), (1, 0)]


def test_box_state_roundtrip():
    box = np.array([100.0, 50.0, 180.0, 110.0])
    assert np.allclose(x_to_box(box_to_z(box)), box)


def moving_box(t: int, vx: float = 20.0) -> np.ndarray:
    return np.array([[100 + vx * t, 300, 180 + vx * t, 360]], float)


def test_sort_keeps_id_through_a_gap():
    tr = Sort(max_age=5, min_hits=3, coast=True)
    ids = []
    for t in range(15):
        dets = np.zeros((0, 4)) if 6 <= t <= 8 else moving_box(t)  # 3 frames without detection
        out = tr.update(dets)
        ids.extend(i for i, _ in out)
    assert set(ids) == {1}
    # while coasting, the predicted box moved with the object
    tr2 = Sort(max_age=5, min_hits=3, coast=True)
    for t in range(6):
        tr2.update(moving_box(t))
    (_, pred), = tr2.update(np.zeros((0, 4)))
    assert abs(pred[0] - moving_box(6)[0, 0]) < 10


def test_after_missed_detections_sort_keeps_id_baseline_does_not():
    # 25 px per frame on an 80 px wide box: consecutive boxes overlap enough for both trackers.
    # After 2 missed frames the object has moved 75 px: the baseline compares with the stale box
    # (IoU ~0.03 -> new ID), SORT compares with its prediction (moved with the learned velocity).
    base, sort = IouTracker(min_hits=1), Sort(min_hits=1)
    base_ids, sort_ids = set(), set()
    for t in range(12):
        dets = np.zeros((0, 4)) if t in (6, 7) else moving_box(t, vx=25)
        base_ids |= {i for i, _ in base.update(dets)}
        sort_ids |= {i for i, _ in sort.update(dets)}
    assert len(base_ids) == 2
    assert sort_ids == {1}


def test_new_track_cannot_follow_a_very_fast_object():
    # Known SORT limitation: a new track starts with zero velocity, so an object moving
    # almost a box width per frame never gets matched and gets a new ID every frame.
    sort = Sort(min_hits=1)
    ids = set()
    for t in range(5):
        ids |= {i for i, _ in sort.update(moving_box(t, vx=70))}
    assert len(ids) == 5


def test_min_hits_suppresses_single_false_positive():
    tr = Sort(min_hits=3)
    for _ in range(5):
        tr.update(np.zeros((0, 4)))
    out = tr.update(np.array([[10, 10, 50, 40]], float))  # appears once
    assert out == []
