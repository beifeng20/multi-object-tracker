"""Kalman filter for one bounding box (as in SORT, Bewley et al. 2016), implemented with numpy.

State: [cx, cy, s, r, vcx, vcy, vs]
  cx, cy = box center, s = area, r = aspect ratio (w/h, assumed constant), v* = velocities per frame.
Measurement: [cx, cy, s, r].
"""
import numpy as np


def box_to_z(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box[:4]
    w, h = x2 - x1, y2 - y1
    return np.array([x1 + w / 2, y1 + h / 2, w * h, w / max(h, 1e-6)])


def x_to_box(x: np.ndarray) -> np.ndarray:
    cx, cy, s, r = x[:4]
    s = max(s, 1e-6)
    w = np.sqrt(s * r)
    h = s / max(w, 1e-6)
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


class KalmanBox:
    # constant velocity for cx, cy, s; r constant
    F = np.eye(7)
    F[0, 4] = F[1, 5] = F[2, 6] = 1.0
    H = np.eye(4, 7)
    R = np.diag([1.0, 1.0, 10.0, 0.01])                          # measurement noise
    Q = np.diag([1.0, 1.0, 1.0, 1e-4, 0.01, 0.01, 1e-4])          # process noise

    def __init__(self, box: np.ndarray):
        self.x = np.zeros(7)
        self.x[:4] = box_to_z(box)
        # position known from the detection, velocity unknown -> large variance
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])

    def predict(self) -> np.ndarray:
        if self.x[2] + self.x[6] <= 0:  # area must stay positive
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return x_to_box(self.x)

    def update(self, box: np.ndarray) -> None:
        y = box_to_z(box) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    @property
    def box(self) -> np.ndarray:
        return x_to_box(self.x)
