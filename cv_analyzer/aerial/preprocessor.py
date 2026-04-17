"""Image loading, normalization, year extraction, and alignment."""

import re
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def extract_year(filename: str) -> int | None:
    """Extract a 4-digit year (1900-2099) from a filename."""
    match = re.search(r"(19|20)\d{2}", filename)
    return int(match.group()) if match else None


def load_images_from_dir(directory: str | Path) -> list[dict]:
    """Load all images from a directory, sorted by year.

    Returns list of dicts: {"path", "filename", "year", "image"}.
    Skips files without a parseable year.
    """
    directory = Path(directory)
    entries = []

    for p in sorted(directory.iterdir()):
        if p.suffix.lower() not in SUPPORTED_EXTS:
            continue
        year = extract_year(p.name)
        if year is None:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        entries.append({
            "path": str(p),
            "filename": p.name,
            "year": year,
            "image": img,
        })

    entries.sort(key=lambda e: e["year"])
    return entries


def normalize(img: np.ndarray, target_long_edge: int = 1024) -> np.ndarray:
    """Resize so longest edge = target, preserving aspect ratio."""
    h, w = img.shape[:2]
    if max(h, w) <= target_long_edge:
        return img
    scale = target_long_edge / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def equalize(gray: np.ndarray) -> np.ndarray:
    """CLAHE histogram equalization for robust cross-decade comparison."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def align_images(
    ref: np.ndarray, target: np.ndarray, max_features: int = 5000
) -> tuple[np.ndarray | None, float]:
    """Align target image to ref using feature matching + homography.

    Returns (warped_target, match_quality).
    match_quality is the inlier ratio (0-1). Returns (None, 0) on failure.
    """
    gray_ref = equalize(to_gray(ref))
    gray_tgt = equalize(to_gray(target))

    # Resize target to match ref dimensions for comparison
    h, w = gray_ref.shape[:2]
    gray_tgt_resized = cv2.resize(gray_tgt, (w, h))
    target_resized = cv2.resize(target, (w, h))

    # Try AKAZE first (more robust to scale/illumination), fall back to ORB
    for detector_name in ("akaze", "orb"):
        if detector_name == "akaze":
            detector = cv2.AKAZE_create()
        else:
            detector = cv2.ORB_create(nfeatures=max_features)

        kp1, des1 = detector.detectAndCompute(gray_ref, None)
        kp2, des2 = detector.detectAndCompute(gray_tgt_resized, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            continue

        if detector_name == "akaze":
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        try:
            raw_matches = matcher.knnMatch(des1, des2, k=2)
        except cv2.error:
            continue

        # Lowe's ratio test
        good = []
        for m_pair in raw_matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        if len(good) < 10:
            continue

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if H is None:
            continue

        inlier_ratio = mask.ravel().sum() / len(mask) if mask is not None else 0
        if inlier_ratio < 0.15:
            continue

        warped = cv2.warpPerspective(target_resized, H, (w, h))
        return warped, float(inlier_ratio)

    # Fallback: simple resize without alignment
    return target_resized, 0.0
