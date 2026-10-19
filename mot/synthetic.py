"""Synthetic tracking scenes with ground truth, to measure trackers with standard MOT metrics.

Objects (car-sized boxes) drive across a 1280x720 frame with constant velocity plus small random acceleration.
The detector is imperfect: box noise, missed detections, short occlusions, and false positives.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class SceneConfig:
    frames: int = 300
    n_objects: int = 12
    box_noise_px: float = 4.0       # std of detection corner noise
    p_miss: float = 0.10            # independent missed detection per frame
    p_occlusion: float = 0.01       # per object per frame: start an occlusion
    occlusion_len: tuple = (3, 8)   # frames the object stays undetected
    false_pos_per_frame: float = 0.5
    speed_px: tuple = (3.0, 10.0)   # pixels per frame
    width, height = 1280, 720


def make_scene(cfg: SceneConfig, seed: int):
    """Returns (ground_truth, detections), each a list per frame.

    ground_truth[f] = list of (object_id, box); detections[f] = (K, 4) array of boxes.
    """
    rng = np.random.default_rng(seed)
    W, H = cfg.width, cfg.height
    objs = []
    for i in range(cfg.n_objects):
        w = rng.uniform(60, 160)
        h = w * rng.uniform(0.6, 0.9)
        start = int(rng.integers(0, cfg.frames // 2))
        side = rng.integers(0, 2)  # enter from left or right, drive across, so paths cross
        x = -w if side == 0 else W
        y = rng.uniform(100, H - 100 - h)
        speed = rng.uniform(*cfg.speed_px) * (1 if side == 0 else -1)
        objs.append({"id": i + 1, "w": w, "h": h, "start": start, "x": x, "y": y,
                     "vx": speed, "vy": rng.normal(0, 0.8), "occluded": 0})
    gt, dets = [], []
    for f in range(cfg.frames):
        gt_f, det_f = [], []
        for o in objs:
            if f < o["start"]:
                continue
            o["vx"] += rng.normal(0, 0.05)
            o["vy"] += rng.normal(0, 0.05)
            o["x"] += o["vx"]
            o["y"] += o["vy"]
            box = np.array([o["x"], o["y"], o["x"] + o["w"], o["y"] + o["h"]])
            if box[2] < 0 or box[0] > W or box[3] < 0 or box[1] > H:
                continue  # outside the frame
            gt_f.append((o["id"], box))
            if o["occluded"] > 0:
                o["occluded"] -= 1
                continue
            if rng.random() < cfg.p_occlusion:
                o["occluded"] = int(rng.integers(*cfg.occlusion_len))
                continue
            if rng.random() < cfg.p_miss:
                continue
            det_f.append(box + rng.normal(0, cfg.box_noise_px, 4))
        for _ in range(rng.poisson(cfg.false_pos_per_frame)):
            w = rng.uniform(40, 120)
            x, y = rng.uniform(0, W - w), rng.uniform(0, H - w)
            det_f.append(np.array([x, y, x + w, y + w * 0.7]))
        gt.append(gt_f)
        dets.append(np.array(det_f).reshape(-1, 4))
    return gt, dets
