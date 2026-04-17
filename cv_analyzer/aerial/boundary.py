"""Detect the green property boundary rectangle on ERIS aerial photos.

ERIS packages overlay a green rectangle marking the subject property.
Detecting it lets us classify findings into Subject / Adjoining / Surrounding.
"""

import cv2
import numpy as np


def detect_green_boundary(img: np.ndarray) -> np.ndarray | None:
    """Find the green property boundary rectangle.

    Returns the contour (Nx1x2 array) of the boundary, or None if not found.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Green overlay: fairly saturated green, mid-to-high value
    # Cast a wide net — ERIS uses various shades of green
    lower = np.array([35, 50, 50])
    upper = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # The boundary is a thin outline, not a filled shape.
    # Dilate to connect any broken segments, then find contours.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = img.shape[:2]
    total_area = h * w

    # The property boundary is typically one of the larger green contours
    # and roughly rectangular. Filter by size and shape.
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Boundary should be between 1% and 60% of image area
        if area < total_area * 0.01 or area > total_area * 0.6:
            continue
        # Check rectangularity
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

    # Return the largest qualifying contour
    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates[0][1]


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

    # pointPolygonTest: >0 inside, =0 on edge, <0 outside
    dist = cv2.pointPolygonTest(boundary_contour, point, measureDist=True)

    if dist >= 0:
        return "subject"

    # "Adjoining" = within a margin of the boundary
    # Use a percentage of the boundary's bounding rect diagonal
    bx, by, bw, bh = cv2.boundingRect(boundary_contour)
    diag = np.sqrt(bw ** 2 + bh ** 2)
    margin = diag * adjoining_margin_pct

    if abs(dist) <= margin:
        return "adjoining"

    return "surrounding"
