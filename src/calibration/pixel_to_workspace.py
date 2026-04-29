import math
import pickle
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def load_lut(lut_path: str):
    with open(lut_path, "rb") as f:
        return pickle.load(f)


class PixelToWorkspaceMapper:
    def __init__(self, lut_path: str, clamp_to_workspace: bool = False, x_min: float = 0.05, x_max: float = 0.95, y_min: float = 0.05, y_max: float = 0.95):
        self.lut = load_lut(lut_path)
        self.clamp_to_workspace = clamp_to_workspace
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)

    def convert_one(self, u: float, v: float) -> Tuple[float, float]:
        x, y = self.lut.pixel_to_workspace(float(u), float(v))
        x = float(x)
        y = float(y)
        if self.clamp_to_workspace:
            x = float(np.clip(x, self.x_min, self.x_max))
            y = float(np.clip(y, self.y_min, self.y_max))
        return x, y

    def convert_many(self, uv_points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
        return [self.convert_one(u, v) for u, v in uv_points]

    def convert_one_safe(self, u: float, v: float):
        try:
            x, y = self.convert_one(u, v)
            if math.isnan(x) or math.isnan(y):
                return {"ok": False, "x": None, "y": None, "error": "nan_output"}
            return {"ok": True, "x": x, "y": y, "error": None}
        except Exception as e:
            return {"ok": False, "x": None, "y": None, "error": str(e)}
