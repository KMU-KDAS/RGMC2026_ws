import cv2
import numpy as np


class SimpleGripperDetector:
    def __init__(self, lower, upper, kernel_size=5, min_area=100):
        self.lower = np.array(lower, dtype=np.uint8)
        self.upper = np.array(upper, dtype=np.uint8)
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.min_area = min_area

    def predict(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"ok": False, "center_uv": None, "mask": mask}

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < self.min_area:
            return {"ok": False, "center_uv": None, "mask": mask}

        M = cv2.moments(largest)
        if M["m00"] == 0:
            return {"ok": False, "center_uv": None, "mask": mask}

        u = M["m10"] / M["m00"]
        v = M["m01"] / M["m00"]

        return {"ok": True, "center_uv": (u, v), "mask": mask}


class YOLOGripperDetector:
    """
    Segmentation-model-based gripper detector for calibration.

    Expected return format is intentionally compatible with SimpleGripperDetector:
        {
            "ok": bool,
            "center_uv": (u, v) or None,
            "mask": np.ndarray or None,
            "contour": np.ndarray or None,
            "score": float or None,
            "bbox_xyxy": (x1, y1, x2, y2) or None,
        }

    Notes
    -----
    - This detector expects a segmentation-capable Ultralytics YOLO model.
    - The model should be trained on the SAME image frame convention used at runtime.
      If your calibration pipeline now uses "undistorted only", the model should ideally
      also have been trained on "undistorted only" images.
    """

    def __init__(
        self,
        model_path,
        conf=0.25,
        min_area=100,
        target_class=None,
        prefer_center=True,
    ):
        """
        Parameters
        ----------
        model_path : str
            Path to Ultralytics YOLO .pt model.
        conf : float
            Minimum confidence threshold.
        min_area : float
            Minimum contour/mask area in pixels.
        target_class : int | None
            If not None, only detections of this class id are considered.
        prefer_center : bool
            If True, when selecting among candidates, slightly prefer detections
            closer to the image center. Useful when only one gripper is expected.
        """
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is not installed. Install it first, e.g. "
                "`pip install ultralytics` in the active environment."
            ) from e

        self.model = YOLO(model_path)
        self.conf = float(conf)
        self.min_area = float(min_area)
        self.target_class = target_class
        self.prefer_center = prefer_center

    def _mask_and_contour_from_polygon(self, image_shape, polygon_xy):
        """
        Build a binary mask and contour from a polygon returned by Ultralytics.

        Parameters
        ----------
        image_shape : tuple
            image.shape
        polygon_xy : np.ndarray
            Nx2 polygon array in pixel coordinates

        Returns
        -------
        mask : np.ndarray
            Binary uint8 mask
        contour : np.ndarray
            OpenCV contour of shape (N,1,2), dtype=int32
        area : float
            Contour area in pixels
        """
        h, w = image_shape[:2]
        poly = np.asarray(polygon_xy, dtype=np.float32)

        if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2:
            return None, None, 0.0

        # clip to valid image bounds
        poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)

        contour = np.round(poly).astype(np.int32).reshape(-1, 1, 2)

        area = cv2.contourArea(contour)
        if area <= 0:
            return None, None, 0.0

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], contourIdx=-1, color=255, thickness=-1)

        return mask, contour, float(area)

    def _centroid_from_mask(self, mask):
        """
        Compute centroid from binary mask moments.

        Returns
        -------
        (u, v) or None
        """
        M = cv2.moments(mask, binaryImage=True)
        if M["m00"] == 0:
            return None
        u = M["m10"] / M["m00"]
        v = M["m01"] / M["m00"]
        return (float(u), float(v))

    def _candidate_rank_score(self, score, center_uv, image_shape):
        """
        Ranking score for choosing among multiple detections.

        Base term: detection confidence
        Optional bonus: closer to image center is slightly preferred
        """
        rank = float(score)

        if self.prefer_center and center_uv is not None:
            h, w = image_shape[:2]
            cx = (w - 1) / 2.0
            cy = (h - 1) / 2.0
            du = center_uv[0] - cx
            dv = center_uv[1] - cy
            dist = np.hypot(du, dv)

            # normalize by image diagonal
            diag = np.hypot(w, h) + 1e-6
            dist_norm = dist / diag

            # small center preference so confidence still dominates
            rank += 0.05 * (1.0 - dist_norm)

        return rank

    def predict(self, image):
        """
        Run segmentation model and return calibration-compatible detection dict.
        """
        if image is None:
            return {
                "ok": False,
                "center_uv": None,
                "mask": None,
                "contour": None,
                "score": None,
                "bbox_xyxy": None,
            }

        try:
            results = self.model(image, verbose=False)
        except Exception:
            return {
                "ok": False,
                "center_uv": None,
                "mask": None,
                "contour": None,
                "score": None,
                "bbox_xyxy": None,
            }

        best = None

        for result in results:
            if result.masks is None:
                continue
            if result.boxes is None:
                continue

            masks_xy = result.masks.xy
            boxes = result.boxes

            # convert tensors safely
            confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else None
            clses = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else None
            xyxy = boxes.xyxy.detach().cpu().numpy() if boxes.xyxy is not None else None

            num = len(masks_xy)
            for i in range(num):
                score = float(confs[i]) if confs is not None and i < len(confs) else 0.0
                cls_id = int(clses[i]) if clses is not None and i < len(clses) else None
                bbox = tuple(map(float, xyxy[i])) if xyxy is not None and i < len(xyxy) else None

                if score < self.conf:
                    continue

                if self.target_class is not None and cls_id != self.target_class:
                    continue

                mask, contour, area = self._mask_and_contour_from_polygon(
                    image.shape, masks_xy[i]
                )
                if mask is None or contour is None:
                    continue

                if area < self.min_area:
                    continue

                center_uv = self._centroid_from_mask(mask)
                if center_uv is None:
                    continue

                rank = self._candidate_rank_score(score, center_uv, image.shape)

                candidate = {
                    "ok": True,
                    "center_uv": center_uv,
                    "mask": mask,
                    "contour": contour,
                    "score": score,
                    "bbox_xyxy": bbox,
                    "class_id": cls_id,
                    "area": area,
                    "rank": rank,
                }

                if best is None or candidate["rank"] > best["rank"]:
                    best = candidate

        if best is None:
            return {
                "ok": False,
                "center_uv": None,
                "mask": None,
                "contour": None,
                "score": None,
                "bbox_xyxy": None,
            }

        return best