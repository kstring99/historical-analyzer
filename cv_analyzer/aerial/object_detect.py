"""Shape-based object detection and color anomaly detection in aerial photos."""

import cv2
import numpy as np

from cv_analyzer.aerial.preprocessor import to_gray


# Target classes for detected objects
CLASSES = {
    "circular_structure": "Circular feature consistent with UST/AST",
    "rectangular_isolated": "Rectangular isolated structure consistent with AST/pad",
    "irregular_pond": "Irregular low-contrast area consistent with lagoon/pit",
    "vegetation_stress": "Anomalous color pattern suggesting vegetation stress",
    "surface_staining": "Dark discoloration suggesting surface staining",
    "industrial_complex": "Cluster of structures with industrial characteristics",
}


def detect_objects(
    img: np.ndarray,
    min_area_pct: float = 0.0005,
    max_area_pct: float = 0.15,
) -> list[dict]:
    """Detect potential environmental features via shape analysis.

    Returns list of detections with class, bbox, confidence, crop.
    """
    h, w = img.shape[:2]
    total_area = h * w
    min_area = total_area * min_area_pct
    max_area = total_area * max_area_pct

    gray = to_gray(img)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Multi-scale edge detection
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)
        bbox_area = cw * ch
        if bbox_area == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        extent = area / bbox_area  # how much of bounding box is filled
        aspect = max(cw, ch) / max(min(cw, ch), 1)

        detection = None

        # Circular structures → tanks
        if circularity > 0.65 and aspect < 1.4:
            diameter_px = np.sqrt(area / np.pi) * 2
            confidence = min(1.0, circularity * 1.2)
            detection = {
                "class": "circular_structure",
                "confidence": round(confidence, 2),
                "description": f"Circular feature (~{int(diameter_px)}px diameter)",
                "metrics": {"circularity": round(circularity, 3), "diameter_px": round(diameter_px, 1)},
            }

        # Rectangular isolated structures → ASTs, pads, buildings
        elif extent > 0.75 and aspect < 3.0 and circularity < 0.6:
            confidence = min(1.0, extent * 0.9)
            # Check if it's isolated (not touching many other contours)
            isolation = _check_isolation(cnt, contours, h, w)
            if isolation > 0.3:
                confidence *= 1.0 + isolation * 0.3
                confidence = min(1.0, confidence)
                detection = {
                    "class": "rectangular_isolated",
                    "confidence": round(confidence, 2),
                    "description": f"Rectangular structure ({cw}x{ch}px, isolation={isolation:.2f})",
                    "metrics": {"extent": round(extent, 3), "aspect": round(aspect, 2), "isolation": round(isolation, 2)},
                }

        # Irregular low-contrast areas → lagoons, pits
        elif circularity < 0.4 and area > min_area * 5:
            region = gray[y : y + ch, x : x + cw]
            if region.size > 0:
                std = float(np.std(region))
                mean_val = float(np.mean(region))
                # Low internal contrast + dark = potential lagoon/pit
                if std < 30 and mean_val < 120:
                    confidence = max(0.3, min(0.85, (30 - std) / 30 * 0.5 + (120 - mean_val) / 120 * 0.5))
                    detection = {
                        "class": "irregular_pond",
                        "confidence": round(confidence, 2),
                        "description": f"Irregular low-contrast area ({cw}x{ch}px)",
                        "metrics": {"internal_std": round(std, 1), "mean_intensity": round(mean_val, 1)},
                    }

        if detection is not None:
            pad = 10
            cx = max(0, x - pad)
            cy = max(0, y - pad)
            crop = img[cy : min(h, y + ch + pad), cx : min(w, x + cw + pad)]
            detection["bbox"] = {"x": int(x), "y": int(y), "w": int(cw), "h": int(ch)}
            detection["area_pct"] = round(area / total_area * 100, 4)
            detection["crop"] = crop
            detections.append(detection)

    # Add color anomaly detections
    detections.extend(_detect_color_anomalies(img, min_area, max_area))

    # Deduplicate overlapping detections (keep higher confidence)
    detections = _nms(detections, iou_threshold=0.4)
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    return detections


def _check_isolation(cnt, all_contours, h: int, w: int, margin_px: int = 50) -> float:
    """Measure how isolated a contour is from its neighbors. Returns 0-1."""
    x, y, cw, ch = cv2.boundingRect(cnt)
    cx, cy = x + cw // 2, y + ch // 2

    nearby = 0
    for other in all_contours:
        if other is cnt:
            continue
        ox, oy, ow, oh = cv2.boundingRect(other)
        ocx, ocy = ox + ow // 2, oy + oh // 2
        dist = np.sqrt((cx - ocx) ** 2 + (cy - ocy) ** 2)
        if dist < margin_px:
            nearby += 1

    # Fewer neighbors = more isolated
    return max(0.0, 1.0 - nearby / 10.0)


def _detect_color_anomalies(
    img: np.ndarray, min_area: float, max_area: float
) -> list[dict]:
    """Detect staining and vegetation stress via HSV analysis."""
    if len(img.shape) < 3:
        return []

    h, w = img.shape[:2]
    total_area = h * w
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    detections = []

    # --- Surface staining: dark, low-saturation patches ---
    # Staining appears as dark brown/black areas with low saturation
    dark_mask = cv2.inRange(hsv, np.array([0, 0, 10]), np.array([180, 80, 80]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area * 3 or area > max_area:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        # Staining is typically irregularly shaped
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity > 0.7:
            continue  # too circular — probably a shadow or structure, not staining

        region_hsv = hsv[y : y + ch, x : x + cw]
        mean_sat = float(np.mean(region_hsv[:, :, 1]))
        mean_val = float(np.mean(region_hsv[:, :, 2]))

        confidence = max(0.25, min(0.8, (80 - mean_sat) / 80 * 0.4 + (80 - mean_val) / 80 * 0.4))
        pad = 10
        crop = img[max(0, y - pad) : min(h, y + ch + pad), max(0, x - pad) : min(w, x + cw + pad)]
        detections.append({
            "class": "surface_staining",
            "confidence": round(confidence, 2),
            "description": f"Dark low-saturation area ({cw}x{ch}px)",
            "metrics": {"mean_saturation": round(mean_sat, 1), "mean_value": round(mean_val, 1)},
            "bbox": {"x": int(x), "y": int(y), "w": int(cw), "h": int(ch)},
            "area_pct": round(area / total_area * 100, 4),
            "crop": crop,
        })

    # --- Vegetation stress: yellow/brown patches in green surroundings ---
    # Green vegetation: H 35-85, S > 40, V > 40
    # Stressed vegetation: H 15-35, S > 30, V > 50 (yellow-brown)
    green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    green_pct = np.sum(green_mask > 0) / total_area

    # Only look for stress if there's significant vegetation in the image
    if green_pct > 0.1:
        stress_mask = cv2.inRange(hsv, np.array([15, 30, 50]), np.array([35, 255, 255]))
        # Only flag stress patches that are surrounded by green
        # Dilate green mask and intersect with stress
        green_dilated = cv2.dilate(green_mask, kernel, iterations=3)
        stress_in_green = cv2.bitwise_and(stress_mask, green_dilated)
        stress_in_green = cv2.morphologyEx(stress_in_green, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(stress_in_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area * 2 or area > max_area:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            confidence = min(0.75, area / total_area * 500)
            pad = 15
            crop = img[max(0, y - pad) : min(h, y + ch + pad), max(0, x - pad) : min(w, x + cw + pad)]
            detections.append({
                "class": "vegetation_stress",
                "confidence": round(confidence, 2),
                "description": f"Yellow-brown patch in green area ({cw}x{ch}px)",
                "metrics": {"surrounding_green_pct": round(green_pct * 100, 1)},
                "bbox": {"x": int(x), "y": int(y), "w": int(cw), "h": int(ch)},
                "area_pct": round(area / total_area * 100, 4),
                "crop": crop,
            })

    return detections


def _nms(detections: list[dict], iou_threshold: float = 0.4) -> list[dict]:
    """Non-maximum suppression: remove overlapping lower-confidence detections."""
    if not detections:
        return detections

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    keep = []

    for det in detections:
        overlaps = False
        for kept in keep:
            if _iou(det["bbox"], kept["bbox"]) > iou_threshold:
                overlaps = True
                break
        if not overlaps:
            keep.append(det)

    return keep


def _iou(a: dict, b: dict) -> float:
    """Intersection over union of two bboxes."""
    x1 = max(a["x"], b["x"])
    y1 = max(a["y"], b["y"])
    x2 = min(a["x"] + a["w"], b["x"] + b["w"])
    y2 = min(a["y"] + a["h"], b["y"] + b["h"])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = a["w"] * a["h"]
    area_b = b["w"] * b["h"]
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0
