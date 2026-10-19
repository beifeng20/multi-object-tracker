"""Track vehicles in a video: YOLO detections per frame -> SORT -> boxes with persistent IDs and trails.

    python track_video.py --input data/project_video.mp4 --output output/tracked.mp4
"""
import argparse
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from mot.trackers import Sort

VEHICLES = {"car", "truck", "bus", "motorcycle"}


def color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--conf", type=float, default=0.4)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    classes = [i for i, n in model.names.items() if n in VEHICLES]
    tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3, coast=True)

    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    trails = defaultdict(lambda: deque(maxlen=30))
    lengths = defaultdict(int)
    t_det = t_trk = 0.0
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        res = model.predict(frame, conf=args.conf, classes=classes, verbose=False)[0]
        dets = res.boxes.xyxy.cpu().numpy()
        t1 = time.perf_counter()
        tracks = tracker.update(dets)
        t2 = time.perf_counter()
        t_det += t1 - t0
        t_trk += t2 - t1
        for tid, box in tracks:
            x1, y1, x2, y2 = (int(v) for v in box)
            c = color(tid)
            lengths[tid] += 1
            trails[tid].append(((x1 + x2) // 2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 3)
            cv2.putText(frame, f"ID {tid}", (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
            pts = np.array(trails[tid], np.int32)
            if len(pts) > 1:
                cv2.polylines(frame, [pts], False, c, 2)
        cv2.putText(frame, f"frame {n}  tracks {len(tracks)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(frame)
        n += 1
    writer.release()
    long_tracks = {k: v for k, v in lengths.items() if v >= 25}
    summary = {"frames": n, "track_ids_created": tracker.next_id - 1, "tracks_shown": len(lengths),
               "tracks_longer_than_1s": len(long_tracks),
               "longest_track_frames": max(lengths.values()) if lengths else 0,
               "yolo_ms_per_frame": round(1000 * t_det / max(n, 1), 1),
               "sort_ms_per_frame": round(1000 * t_trk / max(n, 1), 2)}
    Path(args.output).with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
