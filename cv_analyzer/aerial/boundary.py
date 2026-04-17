"""Detect property boundary overlays on ERIS documents.

Aerials use a GREEN rectangle. Topo maps use a RED/DARK RED rectangle.
Detecting these lets us classify findings into Subject / Adjoining / Surrounding.
"""

import cv2
import numpy as np


def detect_boundary(img: np.ndarray, doc_type: str = "aerial") -> np.ndarray | None:
    """Find the property boundary rectangle.

    doc_type: "aerial" (green boundary) or "topo" (red/dark red boundary).
    Returns the contour (Nx1x2 array) of the boundary, or None if not found.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if doc_type == "topo":
        # Red / dark red boundary on topo maps
        # Red wraps around H=0 in HSV, so we need two ranges
        mask_low = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([12, 255, 255]))
        mask_high = cv2.inRange(hsv, np.array([165, 50, 40]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask_low, mask_high)
    else:
        # Green boundary on aerial photos
        mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = img.shape[:2]
    total_area = h * w

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < total_area * 0.01 or area > total_area * 0.6:
            continue
        rect = cv2.minAreaRect(cnt)
        rect_area = rect[1][0] * rect[1][1]
        if rect_area == 0:
            continue
        rectangularity = area / rect_area
        if rectangularity < 0.5:
            continue
        candidates.append((area, cnt))

    if not candidates:
        return None

    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates[0][1]


# Keep old name as alias for backward compat with CLI
detect_green_boundary = lambda img: detect_boundary(img, "aerial")


def classify_zone(
    detection_bbox: dict,
    boundary_contour: np.ndarray | None,
    img_shape: tuple,
    adjoining_margin_pct: float = 0.15,
) -> str:
    """Classify a detection's zone: subject, adjoining, or surrounding.

    If no boundary was detected, everything is 'unknown'.
    """
    if boundary_contour is None:
        return "unknown"

    h, w = img_shape[:2]
    det_cx = detection_bbox["x"] + detection_bbox["w"] // 2
    det_cy = detection_bbox["y"] + detection_bbox["h"] // 2
    point = (det_cx, det_cy)

    dist = cv2.pointPolygonTest(boundary_contour, point, measureDist=True)

    if dist >= 0:
        return "subject"

    bx, by, bw, bh = cv2.boundingRect(boundary_contour)
    diag = np.sqrt(bw ** 2 + bh ** 2)
    margin = diag * adjoining_margin_pct

    if abs(dist) <= margin:
        return "adjoining"

    return "surrounding"
