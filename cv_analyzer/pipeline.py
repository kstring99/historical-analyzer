"""End-to-end analysis pipeline for aerial, topo, and city directory PDFs.

Each analyzer takes a provider (vision LLM) and returns a normalized doc result:

{
    "doc_type": "aerial" | "topo" | "city_directory",
    "filename": str,
    "years_reviewed": [year, ...],
    "metadata": {...},
    "pages": [PageResult, ...],
    "tables": [ZoneTable, ...],   # subject / adjoining / surrounding
}

A ZoneTable is {"zone_label": str, "rows": [{"year_range", "issues_noted",
"observations", "issues_detail"}]}.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from cv_analyzer.llm_provider import LLMProvider
from cv_analyzer.prompts import (
    AERIAL_PROMPT,
    CITY_DIR_EXTRACT_PROMPT,
    CITY_DIR_SYNTHESIS_PROMPT,
    CONSOLIDATION_PROMPT,
    COVER_PAGE_PROMPT,
    FIM_PROMPT,
    SUMMARY_PROMPT,
    TOPO_PROMPT,
)

logger = logging.getLogger(__name__)

ZONE_LABELS = {
    "subject": "Subject Property",
    "adjoining": "Adjoining Properties",
    "surrounding": "Surrounding Properties",
}


# ---------- Page result ----------

@dataclass
class PageResult:
    page_number: int
    year: str = ""
    subject_obs: str = ""
    adjoining_obs: str = ""
    surrounding_obs: str = ""
    issues: dict = field(default_factory=lambda: {"subject": "No", "adjoining": "No", "surrounding": "No"})
    raw_response: str = ""


# ---------- Image helpers ----------

def pil_to_bytes(img: Image.Image, fmt: str = "JPEG", quality: int = 85, max_dim: int = 1568) -> tuple[bytes, str]:
    """Convert PIL to bytes with sensible resize for vision models."""
    if max(img.size) > max_dim:
        img = img.copy()
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt, quality=quality)
    media = "image/jpeg" if fmt.upper() in ("JPEG", "JPG") else "image/png"
    return buf.getvalue(), media


# ---------- Parsing ----------

def parse_three_zone(response: str) -> PageResult:
    result = PageResult(page_number=0)
    for raw in response.split("\n"):
        line = raw.strip()
        if line.startswith("YEAR:"):
            result.year = line.split(":", 1)[1].strip()
        elif line.startswith("SUBJECT_ISSUES:"):
            result.issues["subject"] = line.split(":", 1)[1].strip()
        elif line.startswith("SUBJECT:"):
            result.subject_obs = line.split(":", 1)[1].strip()
        elif line.startswith("ADJOINING_ISSUES:"):
            result.issues["adjoining"] = line.split(":", 1)[1].strip()
        elif line.startswith("ADJOINING:"):
            result.adjoining_obs = line.split(":", 1)[1].strip()
        elif line.startswith("SURROUNDING_ISSUES:"):
            result.issues["surrounding"] = line.split(":", 1)[1].strip()
        elif line.startswith("SURROUNDING:"):
            result.surrounding_obs = line.split(":", 1)[1].strip()
    result.raw_response = response
    return result


def parse_cover(response: str) -> dict:
    meta = {"raw": response}
    for raw in response.split("\n"):
        if ":" not in raw:
            continue
        key, _, val = raw.partition(":")
        key = key.strip().lower()
        val = val.strip()
        if not val:
            continue
        if "address" in key or "location" in key:
            meta["address"] = val
        elif "coordinate" in key or "lat" in key:
            meta["coordinates"] = val
        elif "project" in key and "number" in key:
            meta["project_number"] = val
        elif "project" in key and ("property" in key or "name" in key):
            meta["project_name"] = val
        elif "order" in key:
            meta["order_number"] = val
    return meta


# ---------- Year-range grouping ----------

_YEAR_DIGITS = re.compile(r"\d+")


def _year_sort_key(p: "PageResult") -> tuple:
    """Sort by leading 4-digit year if present, else by the first integer found,
    then fall back to page_number so Image 1, Image 2, Image 10 order correctly."""
    label = (p.year or "").strip()
    match = re.match(r"(19|20)\d{2}", label)
    if match:
        return (0, int(match.group()), p.page_number)
    nums = _YEAR_DIGITS.findall(label)
    if nums:
        return (1, int(nums[0]), p.page_number)
    return (2, 0, p.page_number)


CONCERN_WORDS = {
    "tank", "tanks", "canopy", "staining", "drums", "fuel", "gasoline", "petroleum",
    "contamination", "hazardous", "waste", "disposal", "spill", "ust", "ast",
    "landfill", "quarry", "mine", "tailings", "treatment plant", "dry cleaner",
    "auto repair", "auto body", "salvage", "machine shop",
}


def _obs_similar(a: str, b: str) -> bool:
    wa = set(re.findall(r"\b\w{4,}\b", a.lower()))
    wb = set(re.findall(r"\b\w{4,}\b", b.lower()))
    if not wa or not wb:
        return a.strip() == b.strip()
    return len(wa & wb) / max(len(wa), len(wb)) > 0.55


def group_zone_rows(pages: list[PageResult], zone: str) -> list[dict]:
    """Group consecutive similar years into TableRows for a single zone."""
    pages = [p for p in pages if p.year and p.year.strip()]
    if not pages:
        return []
    pages = sorted(pages, key=_year_sort_key)

    def get_obs(p: PageResult) -> str:
        return {"subject": p.subject_obs, "adjoining": p.adjoining_obs,
                "surrounding": p.surrounding_obs}[zone]

    def get_issue(p: PageResult) -> str:
        return p.issues.get(zone, "No")

    def issue_flag(s: str) -> str:
        return "Yes" if s.lower().startswith("yes") else "No"

    groups = [[pages[0]]]
    for p in pages[1:]:
        prev = groups[-1][-1]
        same_issue = issue_flag(get_issue(p)) == issue_flag(get_issue(prev))
        if same_issue and _obs_similar(get_obs(prev), get_obs(p)):
            groups[-1].append(p)
        else:
            groups.append([p])

    rows = []
    for grp in groups:
        year_range = grp[0].year if len(grp) == 1 else f"{grp[0].year} - {grp[-1].year}"
        # Take the observation from the last page in the group (usually the most detailed)
        obs = get_obs(grp[-1]) or get_obs(grp[0])
        obs = _strip_cv_jargon(obs)
        issue_raw = get_issue(grp[-1])
        flag = issue_flag(issue_raw)
        rows.append({
            "year_range": year_range,
            "issues_noted": flag,
            "observations": obs,
            "issues_detail": issue_raw if flag == "Yes" else "",
        })
    return rows


# Parenthesized group containing px/isolation/confidence — remove whole group
_PAREN_JARGON = re.compile(r"\s*\([^)]*(?:\dpx|isolation\s*=|confidence\s*=)[^)]*\)", re.IGNORECASE)
_PX_RANGE_PATTERN = re.compile(
    r"[,;]?\s*(?:ranging|from|between|approximately|about|around)?\s*"
    r"~?\d+\s*(?:x\s*\d+)?\s*px\s*(?:to|-|–|—)\s*\d+\s*(?:x\s*\d+)?\s*px"
    r"(?:\s+(?:in|across)?\s*\w+)?",
    re.IGNORECASE,
)
_PX_PATTERN = re.compile(
    r"~?\d+\s*(?:x\s*\d+)?\s*px"
    r"(?:\s+(?:diameter|wide|tall|long|across|in\s+\w+))?",
    re.IGNORECASE,
)
_ISOLATION_PATTERN = re.compile(r"\s*,?\s*isolation\s*=\s*[\d.]+\s*", re.IGNORECASE)
_CHARACTERIZED_ORPHAN = re.compile(
    r",?\s*characterized by\s*(?:\s|,|\.|$)", re.IGNORECASE
)
_CV_PHRASES = [
    (re.compile(r"dark low-saturation areas?", re.I), "discolored areas"),
    (re.compile(r"low-saturation", re.I), ""),
    (re.compile(r"low-contrast", re.I), ""),
    (re.compile(r"high-contrast", re.I), ""),
    (re.compile(r"\bcircularity\b[^,.]*", re.I), ""),
    (re.compile(r"\bconfidence\s*=\s*[\d.]+", re.I), ""),
    (re.compile(r"pixel(?:s)? ?in ?dimension", re.I), ""),
]


_SUBJECT_PROPERTY_PATTERN = re.compile(r"\bsubject\s+property\b", re.IGNORECASE)


def _capitalize_subject_property(text: str) -> str:
    """Normalize every reference to the Subject Property to the proper-noun form."""
    if not text:
        return text
    return _SUBJECT_PROPERTY_PATTERN.sub("Subject Property", text)


def _strip_cv_jargon(text: str) -> str:
    """Scrub any remaining CV-layer leakage from observation text."""
    if not text:
        return text
    text = _PAREN_JARGON.sub("", text)
    text = _PX_RANGE_PATTERN.sub("", text)
    text = _PX_PATTERN.sub("", text)
    text = _ISOLATION_PATTERN.sub("", text)
    for pat, replacement in _CV_PHRASES:
        text = pat.sub(replacement, text)
    # Remove orphan "characterized by" that now lacks a clause
    text = _CHARACTERIZED_ORPHAN.sub(" ", text)
    # collapse whitespace, orphan punctuation, empty parentheses, dangling close-parens
    text = re.sub(r"\s*\(\s*\)", "", text)
    text = re.sub(r"(?<![A-Za-z0-9_])\)", "", text)
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    # Strip trailing dangling connectors
    text = re.sub(r"[,;\s]+(?:ranging|from|between)\s*[.,]?\s*$", "", text, flags=re.I).strip()
    text = re.sub(r"[,;\s]+$", "", text).strip()
    # Normalize Subject Property casing
    text = _capitalize_subject_property(text)
    # Ensure terminal period
    if text and text[-1] not in ".!?":
        text += "."
    return text


def build_zone_tables(
    pages: list[PageResult],
    provider: Optional[LLMProvider] = None,
    doc_type: str = "aerial",
) -> list[dict]:
    """Build the three zone tables for a document.

    Pages are first filtered to drop placeholder rows. If a provider with text
    capability is given and there are enough rows to group, the per-year rows
    are consolidated by the LLM; otherwise a heuristic grouping is used.
    """
    pages = _filter_placeholder_pages(pages)
    tables = []
    for zone in ("subject", "adjoining", "surrounding"):
        per_year = _per_year_rows(pages, zone)
        rows = _consolidate_rows(provider, doc_type, zone, per_year) \
            if (provider is not None and len(per_year) > 2) \
            else group_zone_rows(pages, zone)
        tables.append({"zone_label": ZONE_LABELS[zone], "rows": rows})
    return tables


def _filter_placeholder_pages(pages: list[PageResult]) -> list[PageResult]:
    """Drop rows whose year is an "Image N" placeholder when real years exist.

    If ALL pages are placeholders, keep them so the table isn't empty.
    """
    pages = [p for p in pages if p.year and p.year.strip()]
    has_real_year = any(re.match(r"^(19|20)\d{2}", p.year or "") for p in pages)
    if not has_real_year:
        return pages
    # Also drop rows with empty observations across all three zones — these are
    # pages where the LLM couldn't produce structured output.
    kept = []
    for p in pages:
        if re.match(r"^Image\s+\d", p.year or "") and not any(
            getattr(p, f"{z}_obs", "").strip() for z in ("subject", "adjoining", "surrounding")
        ):
            continue
        kept.append(p)
    return kept


def _per_year_rows(pages: list[PageResult], zone: str) -> list[dict]:
    """Build one row per page for a single zone (pre-consolidation)."""
    pages = sorted(pages, key=_year_sort_key)
    rows = []
    for p in pages:
        obs = {"subject": p.subject_obs, "adjoining": p.adjoining_obs,
               "surrounding": p.surrounding_obs}[zone]
        if not obs.strip():
            continue
        issue_raw = p.issues.get(zone, "No")
        flag = "Yes" if issue_raw.lower().startswith("yes") else "No"
        rows.append({
            "year_range": p.year,
            "issues_noted": flag,
            "observations": _strip_cv_jargon(obs),
            "issues_detail": issue_raw if flag == "Yes" else "",
        })
    return rows


def _consolidate_rows(
    provider: LLMProvider,
    doc_type: str,
    zone: str,
    per_year: list[dict],
) -> list[dict]:
    """Ask the LLM to consolidate per-year rows into year ranges."""
    if not per_year:
        return []

    doc_label = {
        "aerial": "aerial photograph",
        "topo": "topographic map",
        "fim": "fire insurance map",
        "city_directory": "city directory",
    }.get(doc_type, doc_type)

    entries = "\n".join(
        f"{r['year_range']} | {r['issues_noted']} | {r['observations']}"
        for r in per_year
    )
    prompt = CONSOLIDATION_PROMPT.format(
        doc_label=doc_label,
        zone_label=ZONE_LABELS[zone],
        entries=entries,
    )

    try:
        response = provider.generate_text(prompt)
    except Exception as e:
        logger.warning(f"Consolidation LLM call failed for {zone}: {e}")
        return _heuristic_group_from_rows(per_year)

    parsed = _parse_consolidation_response(response)
    if not parsed:
        return _heuristic_group_from_rows(per_year)
    return parsed


def _parse_consolidation_response(response: str) -> list[dict]:
    """Parse 'YEAR | FLAG | OBSERVATION' lines from the LLM response."""
    rows = []
    for raw in response.splitlines():
        line = raw.strip().lstrip("-*• ").strip()
        if not line or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|", 2)]
        if len(parts) != 3:
            continue
        year_range, flag_raw, observation = parts
        flag = "Yes" if flag_raw.lower().startswith("yes") else "No"
        obs = _strip_cv_jargon(observation)
        if not obs or not year_range:
            continue
        rows.append({
            "year_range": year_range,
            "issues_noted": flag,
            "observations": obs,
            "issues_detail": observation if flag == "Yes" else "",
        })
    return rows


def _heuristic_group_from_rows(per_year: list[dict]) -> list[dict]:
    """Fallback grouping on already-prepared per-year rows."""
    if not per_year:
        return []
    groups = [[per_year[0]]]
    for row in per_year[1:]:
        prev = groups[-1][-1]
        same_issue = row["issues_noted"] == prev["issues_noted"]
        if same_issue and _obs_similar(prev["observations"], row["observations"]):
            groups[-1].append(row)
        else:
            groups.append([row])
    out = []
    for grp in groups:
        year_range = grp[0]["year_range"] if len(grp) == 1 \
            else f"{grp[0]['year_range']} - {grp[-1]['year_range']}"
        last = grp[-1]
        out.append({
            "year_range": year_range,
            "issues_noted": last["issues_noted"],
            "observations": last["observations"],
            "issues_detail": last["issues_detail"],
        })
    return out


# ---------- CV hint formatting for aerials ----------

def _describe_cv_hints(
    detections: list[dict],
    changes: list[dict],
) -> str:
    """Give the vision model light hints from CV — no px or isolation scores."""
    if not detections and not changes:
        return ""

    friendly = {
        "circular_structure": "possible circular feature",
        "rectangular_isolated": "possible isolated rectangular structure",
        "irregular_pond": "possible irregular low-lying area or pond",
        "surface_staining": "possible discoloration on the ground",
        "vegetation_stress": "possible distressed vegetation",
        "industrial_complex": "cluster of structures that may be industrial",
    }

    det_bits = []
    for d in detections:
        label = friendly.get(d["class"], d["class"].replace("_", " "))
        zone = d.get("zone", "unknown")
        det_bits.append(f"- {label} in the {zone} zone")
    det_bits = det_bits[:8]  # cap

    ch_bits = []
    for c in changes[:5]:
        t = c["type"].replace("_", " ")
        zone = c.get("zone", "unknown")
        ch_bits.append(f"- {t} in the {zone} zone")

    parts = ["An automated computer-vision pass flagged the following for your attention. "
             "These are HINTS — verify them by looking at the image. Ignore any that you can "
             "clearly see are benign (baseball diamond, parking lot, shadow, etc.):"]
    if det_bits:
        parts.append("Flagged features:\n" + "\n".join(det_bits))
    if ch_bits:
        parts.append("Flagged changes from the prior image:\n" + "\n".join(ch_bits))
    return "\n\n".join(parts) + "\n"


# ---------- Aerial analyzer ----------

def analyze_aerial_pdf(
    provider: LLMProvider,
    pdf_path: str,
    pages: list[dict],          # pre-extracted: [{page, year, image(PIL), source_pdf}]
    cv_results: dict,           # {"image_results": [...], "change_results": [...]}
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Build an aerial doc result using vision LLM (preferred) or CV fallback."""
    filename = Path(pdf_path).name

    # Find cover metadata from the first page
    metadata = {}
    if pages and provider.supports_vision:
        cover_img = pages[0]["image"]
        img_bytes, media = pil_to_bytes(cover_img, max_dim=1200)
        try:
            if progress_cb:
                progress_cb(f"Reading cover of {filename}")
            cover_resp = provider.analyze_image(cover_img_bytes := img_bytes, COVER_PAGE_PROMPT, media)
            metadata = parse_cover(cover_resp)
        except Exception as e:
            logger.warning(f"Cover parse failed: {e}")
            metadata = {}

    page_results: list[PageResult] = []
    img_results_by_page = {r["page"]: r for r in cv_results.get("image_results", [])
                           if "page" in r}
    changes_by_page: dict = {}
    for ch_res in cv_results.get("change_results", []):
        changes_by_page.setdefault(ch_res.get("to_page"), []).extend(
            [dict(c, from_page=ch_res.get("from_page")) for c in ch_res.get("changes", [])]
        )

    # Skip the cover page (first page of a composite aerial PDF).
    content_pages = pages[1:] if len(pages) > 1 else pages

    for i, pg in enumerate(content_pages):
        # Prefer year from PDF text, but the LLM will overwrite once vision returns.
        # Without a year, label images by their order (Image 1, 2, ...).
        label = pg.get("year") if pg.get("year") is not None else f"Image {i + 1}"
        if progress_cb:
            progress_cb(f"Analyzing aerial image {i+1}/{len(content_pages)}")

        detections = img_results_by_page.get(pg.get("page"), {}).get("detections", [])
        changes = changes_by_page.get(pg.get("page"), [])
        hints = _describe_cv_hints(detections, changes)

        prompt = AERIAL_PROMPT.format(cv_hints=hints if hints else
                                      "No automated hints for this image — analyze from scratch.\n")

        try:
            img_bytes, media = pil_to_bytes(pg["image"])
            response = provider.analyze_image(img_bytes, prompt, media)
            pr = parse_three_zone(response)
        except NotImplementedError:
            pr = _aerial_static_fallback(label, detections, changes)
        except Exception as e:
            logger.warning(f"Aerial vision failed on {label}: {e}")
            pr = _aerial_static_fallback(label, detections, changes)

        pr.page_number = pg.get("page", i + 2)
        if not pr.year:
            pr.year = str(label)
        page_results.append(pr)

    tables = build_zone_tables(page_results, provider=provider, doc_type="aerial")
    seen = {p.year: p for p in page_results if p.year}
    years = [p.year for p in sorted(seen.values(), key=_year_sort_key)]

    return {
        "doc_type": "aerial",
        "filename": filename,
        "metadata": metadata,
        "years_reviewed": years,
        "pages": page_results,
        "tables": tables,
    }


def _aerial_static_fallback(year, detections: list[dict], changes: list[dict]) -> PageResult:
    """Deterministic, ESA-voice observation built from CV flags.

    Used when no vision LLM is available. Text-only LLMs don't follow the strict
    response format reliably, so we skip the LLM and synthesize directly.
    """
    pr = PageResult(page_number=0, year=str(year))

    concern_classes = {"circular_structure", "surface_staining", "vegetation_stress",
                       "irregular_pond", "industrial_complex"}
    friendly = {
        "circular_structure": "a circular feature that may represent a storage tank",
        "rectangular_isolated": "an isolated rectangular structure",
        "irregular_pond": "an irregular low-lying area that may represent a pond or pit",
        "surface_staining": "an area of discolored ground that may indicate staining",
        "vegetation_stress": "an area of distressed vegetation",
        "industrial_complex": "a cluster of structures with possible industrial character",
    }

    by_zone: dict[str, list[str]] = {"subject": [], "adjoining": [], "surrounding": []}
    concern_by_zone: dict[str, list[str]] = {"subject": [], "adjoining": [], "surrounding": []}

    # Without vision, the CV layer is the only signal — be conservative about
    # what we surface. High confidence only, and drop the HSV-noisy classes
    # (surface_staining, vegetation_stress) unless they're very high confidence.
    HIGH_BAR = {"surface_staining": 0.70, "vegetation_stress": 0.70}
    DEFAULT_BAR = 0.55

    for d in detections:
        z = d.get("zone")
        if z not in by_zone:
            continue
        threshold = HIGH_BAR.get(d["class"], DEFAULT_BAR)
        if d.get("confidence", 0) < threshold:
            continue
        desc = friendly.get(d["class"], d["class"].replace("_", " "))
        by_zone[z].append(desc)
        if d["class"] in concern_classes:
            concern_by_zone[z].append(desc)

    templates = {
        "subject": {
            "empty": "The subject property appears undeveloped or developed with ordinary land use; no environmental features are apparent from the photograph.",
            "with_items": "The subject property appears to contain {items}.",
        },
        "adjoining": {
            "empty": "The adjoining properties appear to consist of ordinary land use; no environmental features are apparent from the photograph.",
            "with_items": "The adjoining properties appear to include {items}.",
        },
        "surrounding": {
            "empty": "The surrounding area appears to consist of ordinary land use typical of the region; no environmental features of concern are apparent.",
            "with_items": "The surrounding area appears to include {items}.",
        },
    }

    def phrase(z: str) -> str:
        items = list(dict.fromkeys(by_zone[z]))
        if not items:
            return templates[z]["empty"]
        joined = "; ".join(items[:3])
        return templates[z]["with_items"].format(items=joined)

    def issue_flag(z: str) -> str:
        if concern_by_zone[z]:
            return f"Yes — {concern_by_zone[z][0]}"
        return "No"

    pr.subject_obs = phrase("subject")
    pr.adjoining_obs = phrase("adjoining")
    pr.surrounding_obs = phrase("surrounding")
    pr.issues = {
        "subject": issue_flag("subject"),
        "adjoining": issue_flag("adjoining"),
        "surrounding": "No",  # surrounding rarely flagged as a concern from CV alone
    }
    return pr


# ---------- Topo analyzer ----------

def analyze_topo_pdfs(
    provider: LLMProvider,
    pages: list[dict],     # [{page, year, image(PIL), source_pdf}]
    shared_metadata: dict,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Analyze one or more topo map pages (each usually its own PDF)."""
    if not provider.supports_vision:
        return _vision_required_result("topo", pages)

    coords = shared_metadata.get("coordinates", "coordinates not available")
    context = ""
    if "address" in shared_metadata:
        context = f"Located near {shared_metadata['address']}."

    page_results: list[PageResult] = []
    content = [p for p in pages if p.get("year") is not None]
    for i, pg in enumerate(content):
        year = pg["year"]
        if progress_cb:
            progress_cb(f"Analyzing topo {year} ({i+1}/{len(content)})")

        prompt = TOPO_PROMPT.format(coordinates=coords, context=context)
        try:
            img_bytes, media = pil_to_bytes(pg["image"])
            response = provider.analyze_image(img_bytes, prompt, media)
            pr = parse_three_zone(response)
        except Exception as e:
            logger.warning(f"Topo vision failed on {year}: {e}")
            pr = PageResult(page_number=pg["page"], year=str(year),
                            subject_obs="Analysis failed for this map.",
                            adjoining_obs="", surrounding_obs="")
        pr.page_number = pg["page"]
        if not pr.year:
            pr.year = str(year)
        page_results.append(pr)

    tables = build_zone_tables(page_results, provider=provider, doc_type="topo")
    seen = {p.year: p for p in page_results if p.year}
    years = [p.year for p in sorted(seen.values(), key=_year_sort_key)]
    return {
        "doc_type": "topo",
        "filename": ", ".join(sorted({p["source_pdf"] for p in content})),
        "metadata": shared_metadata,
        "years_reviewed": years,
        "pages": page_results,
        "tables": tables,
    }


# ---------- Fire insurance map (FIM / Sanborn) analyzer ----------

def analyze_fim_pdfs(
    provider: LLMProvider,
    pages: list[dict],     # [{page, year, image(PIL), source_pdf}]
    shared_metadata: dict,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Analyze fire insurance map pages (each is its own PDF, red boundary)."""
    if not provider.supports_vision:
        return _vision_required_result("fim", pages)

    coords = shared_metadata.get("coordinates", "coordinates not available")
    context = ""
    if "address" in shared_metadata:
        context = f"Located near {shared_metadata['address']}."

    page_results: list[PageResult] = []
    content = [p for p in pages if p.get("year") is not None]
    for i, pg in enumerate(content):
        year = pg["year"]
        if progress_cb:
            progress_cb(f"Analyzing fire insurance map {year} ({i+1}/{len(content)})")

        prompt = FIM_PROMPT.format(coordinates=coords, context=context)
        try:
            img_bytes, media = pil_to_bytes(pg["image"])
            response = provider.analyze_image(img_bytes, prompt, media)
            pr = parse_three_zone(response)
        except Exception as e:
            logger.warning(f"FIM vision failed on {year}: {e}")
            pr = PageResult(page_number=pg["page"], year=str(year),
                            subject_obs="Analysis failed for this map.",
                            adjoining_obs="", surrounding_obs="")
        pr.page_number = pg["page"]
        if not pr.year:
            pr.year = str(year)
        page_results.append(pr)

    tables = build_zone_tables(page_results, provider=provider, doc_type="fim")
    seen = {p.year: p for p in page_results if p.year}
    years = [p.year for p in sorted(seen.values(), key=_year_sort_key)]
    return {
        "doc_type": "fim",
        "filename": ", ".join(sorted({p["source_pdf"] for p in content})),
        "metadata": shared_metadata,
        "years_reviewed": years,
        "pages": page_results,
        "tables": tables,
    }


# ---------- City directory analyzer ----------

def analyze_city_directory_pdf(
    provider: LLMProvider,
    pdf_path: str,
    pages: list[dict],
    subject_address: str,
    adjoining_addresses: list[str],
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """Two-pass city directory: extract raw entries per page, then synthesize."""
    if not provider.supports_vision:
        return _vision_required_result("city_directory", pages)

    filename = Path(pdf_path).name
    raw_responses: list[str] = []
    content = pages[1:] if len(pages) > 1 else pages  # skip cover
    for i, pg in enumerate(content):
        if progress_cb:
            progress_cb(f"Reading directory page {i+1}/{len(content)} of {filename}")
        try:
            img_bytes, media = pil_to_bytes(pg["image"])
            resp = provider.analyze_image(img_bytes, CITY_DIR_EXTRACT_PROMPT, media)
            raw_responses.append(resp)
        except Exception as e:
            logger.warning(f"CD extract failed on page {pg['page']}: {e}")

    if not raw_responses:
        return {
            "doc_type": "city_directory",
            "filename": filename,
            "metadata": {},
            "years_reviewed": [],
            "pages": [],
            "tables": build_zone_tables([]),
        }

    if progress_cb:
        progress_cb(f"Synthesizing city directory entries for {filename}")

    adj_str = "; ".join(adjoining_addresses) if adjoining_addresses else "Not provided"
    synthesis_prompt = CITY_DIR_SYNTHESIS_PROMPT.format(
        subject_address=subject_address or "Not provided",
        adjoining_addresses=adj_str,
        raw_entries="\n\n---\n\n".join(raw_responses),
    )

    try:
        synthesis = provider.generate_text(synthesis_prompt)
    except Exception as e:
        logger.warning(f"CD synthesis failed: {e}")
        synthesis = ""

    # Parse multi-year synthesis
    page_results: list[PageResult] = []
    current = PageResult(page_number=0)
    for raw in synthesis.split("\n"):
        line = raw.strip()
        if line.startswith("YEAR:"):
            if current.year:
                page_results.append(current)
            current = PageResult(page_number=0, year=line.split(":", 1)[1].strip())
        elif line.startswith("SUBJECT_ISSUES:"):
            current.issues["subject"] = line.split(":", 1)[1].strip()
        elif line.startswith("SUBJECT:"):
            current.subject_obs = line.split(":", 1)[1].strip()
        elif line.startswith("ADJOINING_ISSUES:"):
            current.issues["adjoining"] = line.split(":", 1)[1].strip()
        elif line.startswith("ADJOINING:"):
            current.adjoining_obs = line.split(":", 1)[1].strip()
        elif line.startswith("SURROUNDING_ISSUES:"):
            current.issues["surrounding"] = line.split(":", 1)[1].strip()
        elif line.startswith("SURROUNDING:"):
            current.surrounding_obs = line.split(":", 1)[1].strip()
    if current.year:
        page_results.append(current)

    tables = build_zone_tables(page_results, provider=provider, doc_type="city_directory")
    seen = {p.year: p for p in page_results if p.year}
    years = [p.year for p in sorted(seen.values(), key=_year_sort_key)]

    return {
        "doc_type": "city_directory",
        "filename": filename,
        "metadata": {},
        "years_reviewed": years,
        "pages": page_results,
        "tables": tables,
    }


# ---------- Fallback result when vision is unavailable ----------

def _vision_required_result(doc_type: str, pages: list[dict]) -> dict:
    msg = ("This analysis requires a vision-capable LLM. Add an Anthropic or "
           "OpenAI API key in Settings and re-run.")
    tables = [
        {"zone_label": ZONE_LABELS[z],
         "rows": [{"year_range": "—", "issues_noted": "—",
                   "observations": msg, "issues_detail": ""}]}
        for z in ("subject", "adjoining", "surrounding")
    ]
    return {
        "doc_type": doc_type,
        "filename": ", ".join(sorted({p["source_pdf"] for p in pages})) if pages else "",
        "metadata": {},
        "years_reviewed": [],
        "pages": [],
        "tables": tables,
    }


# ---------- Summary ----------

def build_summary(provider: LLMProvider, doc_results: list[dict]) -> str:
    """Cross-reference summary across all doc types.

    If the vision LLM wasn't available and no real years were captured, skip the
    LLM call entirely and emit a deterministic factual paragraph — the LLM tends
    to hallucinate dates when given "unknown years".
    """
    label_map = {
        "aerial": "aerial photographs",
        "topo": "topographic maps",
        "fim": "fire insurance maps",
        "city_directory": "city directory listings",
    }

    parts = []
    any_real_years = False
    any_concerns = False

    for doc in doc_results:
        label = label_map.get(doc["doc_type"], doc["doc_type"])
        years_list = [y for y in doc.get("years_reviewed", []) if y]
        real_years = [y for y in years_list if re.match(r"^(19|20)\d{2}", y)]
        if real_years:
            any_real_years = True
            years_str = _year_range_str(real_years)
            year_phrase = f"from {years_str}" if years_str else "(years not detected)"
        elif years_list:
            year_phrase = f"across {len(years_list)} image(s); years not visible on the source"
        else:
            year_phrase = "(no images analyzed)"

        concerns = []
        for table in doc.get("tables", []):
            for row in table["rows"]:
                if row["issues_noted"] == "Yes":
                    any_concerns = True
                    detail = row.get("issues_detail") or row["observations"]
                    concerns.append(
                        f"  - {table['zone_label']} ({row['year_range']}): {detail}"
                    )

        section = f"**{label.title()}** {year_phrase}:"
        if concerns:
            section += "\n  Environmental concerns identified:\n" + "\n".join(concerns)
        else:
            section += "\n  No environmental concerns identified."
        parts.append(section)

    findings = "\n\n".join(parts) if parts else "No documents analyzed."

    # If we have no real years AND no concerns, skip the LLM — it has nothing to
    # cross-reference and tends to invent history.
    if not any_real_years and not any_concerns:
        return _deterministic_summary(doc_results, label_map)

    prompt = SUMMARY_PROMPT.format(findings=findings)
    try:
        return provider.generate_text(prompt)
    except Exception as e:
        logger.warning(f"Summary generation failed: {e}")
        return findings


def _year_range_str(years: list[str]) -> str:
    digits = sorted({int(m.group()) for y in years for m in [re.match(r"(19|20)\d{2}", y)] if m})
    if not digits:
        return ""
    if len(digits) == 1:
        return str(digits[0])
    return f"{digits[0]} through {digits[-1]}"


def _deterministic_summary(doc_results: list[dict], label_map: dict[str, str]) -> str:
    """Plain factual paragraph when no years or concerns are available."""
    if not doc_results:
        return "No historical documentation was analyzed."

    sources = []
    for doc in doc_results:
        label = label_map.get(doc["doc_type"], doc["doc_type"])
        n = sum(len(t["rows"]) for t in doc.get("tables", [])) or len(doc.get("years_reviewed", []))
        sources.append(f"{label} ({n} entries)")

    sources_str = ", ".join(sources[:-1]) + (", and " + sources[-1] if len(sources) > 1 else sources[0])
    return (
        f"Historical sources reviewed for the subject property included {sources_str}. "
        "No environmental concerns were identified in the historical documentation "
        "reviewed."
    )
