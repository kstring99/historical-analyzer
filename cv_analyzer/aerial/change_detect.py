"""Pairwise change detection between aerial photos from different years."""

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from cv_analyzer.aerial.preprocessor import to_gray, equalize, align_images


def detect_changes(
    img_a: np.ndarray,
    img_b: np.ndarray,
    year_a: int,
    year_b: int,
    min_area_pct: float = 0.001,
    ssim_threshold: float = 0.4,
) -> dict:
    """Detect structural changes between two aerial images.

    img_a is the earlier image, img_b is the later one.
    Returns a dict with change regions, SSIM score, and alignment quality.
    """
    # Align img_b to img_a's coordinate space
    aligned_b, align_quality = align_images(img_a, img_b)
    if aligned_b is None:
        aligned_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
        align_quality = 0.0

    gray_a = equalize(to_gray(img_a))
    gray_b = equalize(to_gray(aligned_b))

    # Ensure same dimensions
    h = min(gray_a.shape[0], gray_b.shape[0])
    w = min(gray_a.shape[1], gray_b.shape[1])
    gray_a = gray_a[:h, :w]
    gray_b = gray_b[:h, :w]

    # Structural similarity — produces a per-pixel similarity map
    ssim_score, ssim_map = structural_similarity(gray_a, gray_b, full=True)
    diff_map = (1.0 - ssim_map) * 255
    diff_map = diff_map.astype(np.uint8)

    # Threshold to find changed regions
    _, thresh = cv2.threshold(diff_map, 80, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_area = h * w
    min_area = total_area * min_area_pct
    changes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)

        # Determine if structure appeared or disappeared by comparing
        # mean intensity in the change region
        region_a = gray_a[y : y + ch, x : x + cw]
        region_b = gray_b[y : y + ch, x : x + cw]
        mean_a = float(np.mean(region_a))
        mean_b = float(np.mean(region_b))

        # Higher contrast/intensity in the later image → something appeared
        # Higher in earlier → something disappeared
        intensity_shift = mean_b - mean_a
        if abs(intensity_shift) < 5:
            change_type = "modified"
        elif intensity_shift > 0:
            change_type = "structure_appeared"
        else:
            change_type = "structure_disappeared"

        # Local SSIM for this region
        if region_a.size > 0 and region_b.size > 0:
            local_ssim = float(structural_similarity(region_a, region_b))
        else:
            local_ssim = 0.0

        # Crop from the later image (for appeared) or earlier (for disappeared)
        source = aligned_b if change_type == "structure_appeared" else img_a
        if len(source.shape) == 2:
            crop = source[y : y + ch, x : x + cw]
        else:
            source_resized = cv2.resize(source, (w, h))
            crop = source_resized[y : y + ch, x : x + cw]

        changes.append({
            "type": change_type,
            "bbox": {"x": int(x), "y": int(y), "w": int(cw), "h": int(ch)},
            "area_pct": round(area / total_area * 100, 3),
            "confidence": round(1.0 - local_ssim, 2),
            "intensity_shift": round(intensity_shift, 1),
            "crop": crop,
        })

    # Sort by area (largest first)
    changes.sort(key=lambda c: c["area_pct"], reverse=True)

    return {
        "from_year": year_a,
        "to_year": year_b,
        "ssim_score": round(float(ssim_score), 3),
        "alignment_quality": round(align_quality, 3),
        "total_changed_pct": round(
            sum(c["area_pct"] for c in changes), 2
        ),
        "changes": changes,
    }
