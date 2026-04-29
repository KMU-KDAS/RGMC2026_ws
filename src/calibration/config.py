import copy
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class CalibrationConfig:
    grid_size: int = 33
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    frames_per_point: int = 5
    settle_time_sec: float = 0.7
    inter_frame_sec: float = 0.05
    reverse_xy: bool = False
    resume: bool = False
    save_masks: bool = True
    write_debug_json: bool = True
    output_subdir: str = "rope_calibration"
    custom_points: Optional[List[Tuple[float, float]]] = None
    camera_params_file: Optional[str] = None
    planar_only: bool = True
    prefer_raw_base_image: bool = True
    min_valid_frames: int = 3
    aggregate_mode: str = "median"
    detector_lower_hsv: Tuple[int, int, int] = (10, 120, 120)
    detector_upper_hsv: Tuple[int, int, int] = (15, 255, 255)
    detector_kernel_size: int = 5
    detector_min_area: int = 100

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_custom_points(value):
    if value is None:
        return None
    return [(float(x), float(y)) for x, y in value]


def load_calibration_config(yaml_path: str) -> Dict[str, Any]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def calibration_config_from_yaml(yaml_path: str) -> CalibrationConfig:
    cfg = load_calibration_config(yaml_path)
    section = copy.deepcopy(cfg.get("rope_calibration", {}))
    section["custom_points"] = _normalize_custom_points(section.get("custom_points"))
    return CalibrationConfig(**{k: section[k] for k in section if k in CalibrationConfig.__annotations__})
