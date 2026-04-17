"""PDF → image extraction with year detection.

Handles two modes:
- Composite PDFs (aerials, city directories): multi-page, one year per page
- Individual PDFs (topos): one PDF per map, year in filename
"""

import re
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image

from cv_analyzer.aerial.preprocessor import extract_year


def extract_composite_pdf(
    pdf_path: str, dpi: int = 200
) -> list[dict]:
    """Extract all pages from a composite PDF (aerial package or city directory).

    Attempts year extraction from embedded text first, then falls back to
    page-order numbering.

    Returns list of {"page": int, "year": int|None, "image": PIL.Image}.
    """
    pages = convert_from_path(pdf_path, dpi=dpi)
    results = []

    # Try pdfplumber for text-based year extraction
    text_years = _extract_years_from_text(pdf_path, len(pages))

    for i, page_img in enumerate(pages):
        year = text_years.get(i)
        results.append({
            "page": i + 1,
            "year": year,
            "image": page_img,
            "source_pdf": Path(pdf_path).name,
        })

    return results


def extract_individual_pdfs(
    pdf_paths: list[str], dpi: int = 200
) -> list[dict]:
    """Extract from individual PDFs (one per topo map). Year from filename."""
    results = []

    for pdf_path in sorted(pdf_paths):
        filename = Path(pdf_path).stem
        year = extract_year(filename)
        pages = convert_from_path(pdf_path, dpi=dpi)

        if pages:
            results.append({
                "page": 1,
                "year": year,
                "image": pages[0],  # topos are single-page
                "source_pdf": Path(pdf_path).name,
            })

    results.sort(key=lambda r: r["year"] or 0)
    return results


def _extract_years_from_text(pdf_path: str, num_pages: int) -> dict[int, int | None]:
    """Try to extract years from PDF text using pdfplumber."""
    years = {}
    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                # Look for 4-digit years in the text
                matches = re.findall(r"\b(19[3-9]\d|20[0-2]\d)\b", text)
                if matches:
                    # Take the most common year on the page, or the last one
                    # (footer years tend to be at the end)
                    years[i] = int(matches[-1])
    except ImportError:
        pass
    except Exception:
        pass

    return years


def pil_to_cv2(pil_img: Image.Image):
    """Convert PIL Image to OpenCV BGR numpy array."""
    import numpy as np

    rgb = pil_img.convert("RGB")
    arr = np.array(rgb)
    return arr[:, :, ::-1].copy()  # RGB → BGR
