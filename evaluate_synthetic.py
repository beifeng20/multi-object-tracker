"""Compare SORT with the motion-free IoU baseline on synthetic scenes (standard CLEAR-MOT / ID metrics).

    python evaluate_synthetic.py --scenes 20
"""
import argparse
import json
from pathlib import Path

import motmetrics as mm
import numpy as np

from mot.association import iou_matrix
from mot.synthetic import SceneConfig, make_scene
from mot.trackers import IouTracker, Sort


def evaluate(tracker_cls, scenes, cfg: SceneConfig, **kw):
    acc_list = []
    for seed in scenes:
        gt, dets = make_scene(cfg, seed)
        tracker = tracker_cls(**kw)
        acc = mm.MOTAccumulator(auto_id=True)
        for gt_f, det_f in zip(gt, dets):
            out = tracker.update(det_f)
            g_ids = [g for g, _ in gt_f]
            t_ids = [t for t, _ in out]
            g_boxes = np.array([b for _, b in gt_f]).reshape(-1, 4)
            t_boxes = np.array([b for _, b in out]).reshape(-1, 4)
            # distance = 1 - IoU; NaN = "cannot match" (IoU below 0.5, the usual MOT criterion)
            iou = iou_matrix(g_boxes, t_boxes)
            dist = np.where(iou >= 0.5, 1.0 - iou, np.nan)
            acc.update(g_ids, t_ids, dist)
        acc_list.append(acc)
    mh = mm.metrics.create()
    summary = mh.compute_many(acc_list, metrics=["mota", "idf1", "num_switches", "num_false_positives",
                                                 "num_misses", "mostly_tracked", "num_unique_objects"],
                              generate_overall=True)
    return summary.loc["OVERALL"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=20)
    args = ap.parse_args()
    setups = {
        "normal speed (3-10 px/frame)": SceneConfig(),
        "fast (15-30 px/frame)": SceneConfig(speed_px=(15.0, 30.0)),
    }
    trackers = {
        "IoU baseline": (IouTracker, {}),
        "IoU baseline + coasting": (IouTracker, {"coast": True}),
        "SORT": (Sort, {}),
        "SORT + coasting": (Sort, {"coast": True}),
    }
    results = {}
    for scene_name, cfg in setups.items():
        results[scene_name] = {}
        for name, (cls, kw) in trackers.items():
            r = evaluate(cls, range(args.scenes), cfg, **kw)
            row = {"MOTA": round(float(r["mota"]), 4), "IDF1": round(float(r["idf1"]), 4),
                   "ID switches": int(r["num_switches"]), "false positives": int(r["num_false_positives"]),
                   "misses": int(r["num_misses"]),
                   "mostly tracked": f"{int(r['mostly_tracked'])}/{int(r['num_unique_objects'])}"}
            results[scene_name][name] = row
            print(f"{scene_name:30s} {name:26s} {row}")
    cfg = SceneConfig()
    Path("results").mkdir(exist_ok=True)
    Path("results/synthetic.json").write_text(json.dumps(
        {"scenes_per_setup": args.scenes, "frames_per_scene": cfg.frames, "objects_per_scene": cfg.n_objects,
         "detector": {"box_noise_px": cfg.box_noise_px, "p_miss": cfg.p_miss, "p_occlusion": cfg.p_occlusion,
                      "occlusion_frames": list(cfg.occlusion_len), "false_pos_per_frame": cfg.false_pos_per_frame},
         "tracker_params": {"max_age": 5, "min_hits": 3, "iou_threshold": 0.3},
         "results": results}, indent=2))


if __name__ == "__main__":
    main()
