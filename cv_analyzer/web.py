"""FastAPI web app for the historical documentation analyzer.

Routes uploaded PDFs by filename → aerial / topo / city_directory, and runs each
type through its own analyzer. Aerial uses the CV pipeline plus a vision LLM for
observations. Topo and city directory use the vision LLM directly.
"""

import asyncio
import concurrent.futures
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from cv_analyzer.aerial.preprocessor import normalize
from cv_analyzer.aerial.boundary import detect_boundary, classify_zone
from cv_analyzer.aerial.change_detect import detect_changes
from cv_analyzer.aerial.object_detect import detect_objects
from cv_analyzer.llm_provider import resolve_provider, LLMProvider
from cv_analyzer.ollama_client import check_available, list_models
from cv_analyzer.pdf_extract import (
    extract_composite_pdf,
    extract_individual_pdfs,
    pil_to_cv2,
)
from cv_analyzer.pipeline import (
    analyze_aerial_pdf,
    analyze_city_directory_pdf,
    analyze_fim_pdfs,
    analyze_topo_pdfs,
    build_summary,
)
from cv_analyzer.report import generate_report

app = FastAPI(title="Historical Documentation Analyzer")

STATIC_DIR = Path(__file__).parent.parent / "static"
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
REPORT_DIR = Path(__file__).parent.parent / "cv_report"
REPORT_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


# ---------- Job model ----------

class JobStatus(str, Enum):
    PENDING = "pending"
    EXTRACTING = "extracting"
    ANALYZING = "analyzing"
    SYNTHESIZING = "synthesizing"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class Job:
    job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: JobStatus = JobStatus.PENDING
    progress: int = 0
    total_pages: int = 0
    current_file: str = ""
    error: str | None = None
    results: list = field(default_factory=list)
    summary: str = ""
    rec_candidates: list = field(default_factory=list)


JOBS: dict[str, Job] = {}


# ---------- Filename classification ----------

def classify_filename(name: str) -> str:
    n = name.lower()
    if "aerial" in n:
        return "aerial"
    if "topo" in n:
        return "topo"
    if "fim" in n or "sanborn" in n or "fire" in n or "insurance" in n:
        return "fim"
    if "cd" in n or "city" in n or "director" in n:
        return "city_directory"
    return "unknown"


# ---------- Routes ----------

@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/foia")
async def foia_placeholder():
    return HTMLResponse("<h1>FOIA Manager</h1><p>Not yet available in this build.</p>")


@app.get("/site-data")
async def site_data_placeholder():
    return HTMLResponse("<h1>Site Data</h1><p>Not yet available in this build.</p>")


@app.get("/api/llm-status")
async def llm_status():
    available = check_available()
    models = list_models() if available else []
    return {"available": available, "models": models}


@app.post("/api/upload")
async def upload(
    files: list[UploadFile] = File(...),
    provider: str = Form(default="auto"),
    api_key: str = Form(default=""),
    subject_address: str = Form(default=""),
    adjoining_addresses: str = Form(default=""),
):
    """Accept a mixed set of PDFs, classify by filename, kick off analysis."""
    job = Job()
    JOBS[job.job_id] = job

    job_dir = UPLOAD_DIR / job.job_id
    job_dir.mkdir(exist_ok=True)

    saved: list[tuple[str, str]] = []  # (path, doc_type)
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(400, f"Only PDF files accepted: {f.filename}")
        dest = job_dir / f.filename
        content = await f.read()
        dest.write_bytes(content)
        saved.append((str(dest), classify_filename(f.filename)))

    # Parse adjoining address block: lines of "address = direction"
    adj_list = []
    for line in (adjoining_addresses or "").splitlines():
        line = line.strip()
        if line:
            adj_list.append(line)

    asyncio.create_task(_run_job(job, saved, provider.strip() or "auto",
                                 (api_key or "").strip(),
                                 subject_address.strip(), adj_list))
    return {"job_id": job.job_id, "files": [f.filename for f in files]}


async def _run_job(
    job: Job,
    saved: list[tuple[str, str]],
    provider_name: str,
    api_key: str,
    subject_address: str,
    adjoining_addresses: list[str],
):
    loop = asyncio.get_event_loop()

    def update(step: str):
        job.current_file = step
        # Single source of truth for progress. Clamp so the bar never exceeds 100%.
        job.progress = min(job.progress + 1, job.total_pages or 1)

    try:
        # Resolve provider once. If no key and auto → ollama (text only).
        try:
            provider = resolve_provider(provider_name, api_key or None)
        except RuntimeError as e:
            job.status = JobStatus.ERROR
            job.error = str(e)
            return

        # Group files by detected type
        aerial_paths = [p for p, t in saved if t == "aerial"]
        topo_paths = [p for p, t in saved if t == "topo"]
        fim_paths = [p for p, t in saved if t == "fim"]
        cd_paths = [p for p, t in saved if t == "city_directory"]
        unknown_paths = [p for p, t in saved if t == "unknown"]
        # Treat unknowns as aerials (most common composite format)
        aerial_paths.extend(unknown_paths)

        # Start at 0 — each stage adds its actual page count after extracting.
        job.total_pages = 0
        job.status = JobStatus.EXTRACTING
        job.current_file = "Extracting pages from PDFs..."

        doc_results: list[dict] = []

        # --- Aerials ---
        for pdf_path in aerial_paths:
            filename = Path(pdf_path).name
            job.current_file = f"Extracting {filename}..."
            extracted = await loop.run_in_executor(_executor, extract_composite_pdf, pdf_path)
            # One update() call per content page (cover is not bumped separately)
            job.total_pages += max(1, len(extracted) - 1)

            # Prepare CV images — skip page 1 (cover). Year may be None on ERIS PDFs;
            # use page order for sorting and pairwise change detection.
            cv_entries = []
            for pg in extracted[1:]:
                cv_img = pil_to_cv2(pg["image"])
                cv_img = normalize(cv_img, target_long_edge=1024)
                cv_entries.append({
                    "year": pg.get("year"),
                    "page": pg["page"],
                    "image": cv_img,
                    "source_pdf": pg["source_pdf"],
                })
            # Already in page order; keep that

            job.status = JobStatus.ANALYZING

            # Run CV: boundary, object detect, change detect
            for e in cv_entries:
                e["boundary"] = detect_boundary(e["image"], "aerial")

            image_results = []
            for i, e in enumerate(cv_entries):
                label = e["year"] if e["year"] is not None else f"p{e['page']}"
                job.current_file = f"Scanning {filename} — image {i+1}/{len(cv_entries)}"
                dets = await loop.run_in_executor(_executor, detect_objects, e["image"])
                b = e.get("boundary")
                for d in dets:
                    d["zone"] = classify_zone(d["bbox"], b, e["image"].shape)
                image_results.append({
                    "filename": f"{e['source_pdf']}_p{e['page']}",
                    "year": e["year"],
                    "page": e["page"],
                    "detections": dets,
                })

            change_results = []
            for i in range(len(cv_entries) - 1):
                a = cv_entries[i]
                b = cv_entries[i + 1]
                job.current_file = f"Comparing {filename} — {i+1}/{max(len(cv_entries)-1, 1)}"
                res = await loop.run_in_executor(
                    _executor, detect_changes,
                    a["image"], b["image"],
                    a["year"] or a["page"], b["year"] or b["page"],
                )
                res["from_page"] = a["page"]
                res["to_page"] = b["page"]
                bnd = b.get("boundary")
                for ch in res.get("changes", []):
                    ch["zone"] = classify_zone(ch["bbox"], bnd, b["image"].shape)
                change_results.append(res)

            # Generate report (REC candidates + crops) off the CV results
            report = await loop.run_in_executor(
                _executor, generate_report,
                filename, str(REPORT_DIR / job.job_id), image_results, change_results,
            )
            job.rec_candidates.extend(report.get("rec_candidates", []))

            # Observation generation via LLM (vision-preferred)
            job.status = JobStatus.SYNTHESIZING
            doc = await loop.run_in_executor(
                _executor, analyze_aerial_pdf,
                provider, pdf_path, extracted,
                {"image_results": image_results, "change_results": change_results},
                update,
            )
            doc_results.append(doc)

        def _aerial_shared_meta() -> dict:
            for d in doc_results:
                if d.get("doc_type") == "aerial" and d.get("metadata"):
                    return d["metadata"]
            return {}

        # --- Topo maps (one PDF per map) ---
        if topo_paths:
            extracted_topos = await loop.run_in_executor(
                _executor, extract_individual_pdfs, topo_paths
            )
            if extracted_topos:
                job.total_pages += len(extracted_topos)
                job.status = JobStatus.ANALYZING
                doc = await loop.run_in_executor(
                    _executor, analyze_topo_pdfs,
                    provider, extracted_topos, _aerial_shared_meta(), update,
                )
                doc_results.append(doc)

        # --- Fire insurance maps (one PDF per map, red boundary) ---
        if fim_paths:
            extracted_fims = await loop.run_in_executor(
                _executor, extract_individual_pdfs, fim_paths
            )
            if extracted_fims:
                job.total_pages += len(extracted_fims)
                job.status = JobStatus.ANALYZING
                doc = await loop.run_in_executor(
                    _executor, analyze_fim_pdfs,
                    provider, extracted_fims, _aerial_shared_meta(), update,
                )
                doc_results.append(doc)

        # --- City directories ---
        for pdf_path in cd_paths:
            filename = Path(pdf_path).name
            job.current_file = f"Extracting {filename}..."
            extracted = await loop.run_in_executor(_executor, extract_composite_pdf, pdf_path)
            # One update() per content page + one for synthesis
            job.total_pages += len(extracted)

            job.status = JobStatus.ANALYZING
            doc = await loop.run_in_executor(
                _executor, analyze_city_directory_pdf,
                provider, pdf_path, extracted,
                subject_address, adjoining_addresses, update,
            )
            doc_results.append(doc)

        # --- Summary ---
        job.status = JobStatus.SYNTHESIZING
        job.current_file = "Writing cross-reference summary..."
        if doc_results:
            summary = await loop.run_in_executor(_executor, build_summary, provider, doc_results)
        else:
            summary = "No documents were analyzed."

        # Serialize results for the frontend (strip non-JSON objects)
        job.results = [_serialize_doc(d) for d in doc_results]
        job.summary = summary
        job.rec_candidates = sorted(job.rec_candidates, key=lambda r: r.get("confidence", 0), reverse=True)
        job.status = JobStatus.COMPLETE
        # Ensure the bar lands on exactly 100% regardless of update() count drift
        job.total_pages = max(job.total_pages, 1)
        job.progress = job.total_pages
        job.current_file = "Complete"

    except Exception as e:
        import traceback
        traceback.print_exc()
        job.status = JobStatus.ERROR
        job.error = str(e)


def _estimate_pages(pdf_path: str) -> int:
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    except Exception:
        return 5  # rough default


def _serialize_doc(doc: dict) -> dict:
    """Drop PageResult objects — frontend only needs tables + metadata."""
    years = doc.get("years_reviewed", [])
    # Collapse "Image 1..N" placeholders into a single readable line so the
    # frontend "Years reviewed" header stays concise.
    if years and all(y.startswith("Image ") for y in years):
        years = [f"{len(years)} image(s); years not visible on the source"]
    return {
        "doc_type": doc["doc_type"],
        "filename": doc.get("filename", ""),
        "metadata": {k: v for k, v in doc.get("metadata", {}).items() if k != "raw"},
        "years_reviewed": years,
        "tables": doc.get("tables", []),
    }


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "progress": job.progress,
        "total_pages": job.total_pages,
        "current_file": job.current_file,
        "error": job.error,
    }


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status == JobStatus.ERROR:
        raise HTTPException(500, job.error or "Analysis failed")
    if job.status != JobStatus.COMPLETE:
        raise HTTPException(202, "Still processing")
    return {
        "results": job.results,
        "summary": job.summary,
        "rec_candidates": job.rec_candidates,
    }


@app.post("/api/export/{job_id}")
async def export_html(job_id: str):
    job = JOBS.get(job_id)
    if not job or job.status != JobStatus.COMPLETE:
        raise HTTPException(404)
    return HTMLResponse(content=_render_standalone_html(job))


def _render_standalone_html(job: Job) -> str:
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>Historical Documentation Review</title>",
        "<style>",
        "body{font-family:Georgia,serif;max-width:960px;margin:40px auto;padding:0 24px;color:#1a1a1a}",
        "h1{font-size:1.6em}",
        "h2{font-size:1.25em;margin-top:2em;border-bottom:1px solid #bbb;padding-bottom:4px}",
        "h3{font-size:1.05em;margin-top:1.5em}",
        "table{width:100%;border-collapse:collapse;margin:8px 0 24px}",
        "th,td{border:1px solid #666;padding:8px;text-align:left;vertical-align:top;font-size:0.95em}",
        "th{background:#e8e8e8}",
        "p.caption{text-align:center;font-weight:bold;margin:4px 0}",
        ".summary{white-space:pre-wrap;line-height:1.55}",
        "</style></head><body>",
        "<h1>Historical Documentation Review</h1>",
    ]

    if job.summary:
        parts.append("<h2>Summary</h2>")
        parts.append(f"<div class='summary'>{_escape_html(job.summary)}</div>")

    label_map = {
        "aerial": ("Aerial Photographs", "AERIAL PHOTOGRAPH SUMMARY"),
        "topo": ("Topographic Maps", "TOPOGRAPHIC MAP SUMMARY"),
        "fim": ("Fire Insurance Maps", "FIRE INSURANCE MAP SUMMARY"),
        "city_directory": ("City Directories", "STREET DIRECTORY SUMMARY"),
    }
    for doc in job.results:
        heading, caption = label_map.get(doc["doc_type"], (doc["doc_type"], doc["doc_type"].upper()))
        parts.append(f"<h2>{heading}</h2>")
        if doc.get("years_reviewed"):
            parts.append(f"<p>Years reviewed: <strong>{', '.join(doc['years_reviewed'])}</strong></p>")
        obs_col = "Occupants" if doc["doc_type"] == "city_directory" else "Observations"
        for table in doc["tables"]:
            parts.append(f"<p class='caption'>{caption} – {table['zone_label']}</p>")
            parts.append("<table><thead><tr><th>Year</th><th>Issues Noted</th>"
                         f"<th>{obs_col}</th></tr></thead><tbody>")
            for row in table["rows"]:
                parts.append(
                    f"<tr><td>{_escape_html(row['year_range'])}</td>"
                    f"<td>{_escape_html(row['issues_noted'])}</td>"
                    f"<td>{_escape_html(row['observations'])}</td></tr>"
                )
            parts.append("</tbody></table>")

    parts.append("</body></html>")
    return "".join(parts)


def _escape_html(s: str) -> str:
    if not s:
        return ""
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
