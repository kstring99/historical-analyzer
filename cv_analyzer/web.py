"""FastAPI web app for the CV-based historical analyzer.

Same UX as the cloud-based app/ — upload PDFs, see progress, get tables.
Everything runs local: OpenCV + ollama, no cloud APIs.
"""

import asyncio
import concurrent.futures
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from cv_analyzer.aerial.preprocessor import normalize
from cv_analyzer.aerial.boundary import detect_boundary, classify_zone
from cv_analyzer.aerial.change_detect import detect_changes
from cv_analyzer.aerial.object_detect import detect_objects
from cv_analyzer.ollama_client import check_available, list_models
from cv_analyzer.pdf_extract import extract_composite_pdf, extract_individual_pdfs, pil_to_cv2
from cv_analyzer.report import generate_report
from cv_analyzer.table_builder import build_tables, format_tables_html, format_tables_text

app = FastAPI(title="Historical Documentation Analyzer (Local CV)")

STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
REPORT_DIR = Path(__file__).parent.parent / "cv_report"
REPORT_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)


class JobStatus(str, Enum):
    PENDING = "pending"
    EXTRACTING = "extracting"
    ANALYZING = "analyzing"
    GENERATING_TABLES = "generating_tables"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class Job:
    job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: JobStatus = JobStatus.PENDING
    progress: int = 0
    total_steps: int = 0
    current_step: str = ""
    error: str | None = None
    tables_html: str = ""
    tables_text: str = ""
    report_json: dict = field(default_factory=dict)
    doc_type: str = "aerial"


JOBS: dict[str, Job] = {}


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/llm-status")
async def llm_status():
    """Check if ollama is available and return model list."""
    available = check_available()
    models = list_models() if available else []
    return {"available": available, "models": models}


@app.post("/api/upload")
async def upload(
    files: list[UploadFile] = File(...),
    doc_type: str = Form(default="aerial"),
    llm_model: str = Form(default="llama3.2:3b"),
):
    """Upload PDFs and start analysis."""
    job = Job(doc_type=doc_type)
    JOBS[job.job_id] = job

    job_dir = UPLOAD_DIR / job.job_id
    job_dir.mkdir(exist_ok=True)

    saved = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(400, f"Only PDF files accepted: {f.filename}")
        dest = job_dir / f.filename
        content = await f.read()
        with open(dest, "wb") as out:
            out.write(content)
        saved.append(str(dest))

    asyncio.create_task(_process_job(job, saved, doc_type, llm_model))
    return {"job_id": job.job_id, "files": [f.filename for f in files]}


async def _process_job(job: Job, pdf_paths: list[str], doc_type: str, llm_model: str):
    loop = asyncio.get_event_loop()
    try:
        # --- Step 1: Extract pages from PDFs ---
        job.status = JobStatus.EXTRACTING
        job.current_step = "Extracting pages from PDFs..."

        if doc_type == "topo":
            pages = await loop.run_in_executor(
                _executor, extract_individual_pdfs, pdf_paths
            )
        else:
            all_pages = []
            for pdf_path in pdf_paths:
                extracted = await loop.run_in_executor(
                    _executor, extract_composite_pdf, pdf_path
                )
                all_pages.extend(extracted)
            pages = all_pages

        if not pages:
            job.status = JobStatus.ERROR
            job.error = "No pages extracted from uploaded PDFs."
            return

        # Convert PIL images to OpenCV format and normalize
        entries = []
        for p in pages:
            cv_img = pil_to_cv2(p["image"])
            cv_img = normalize(cv_img, target_long_edge=1024)
            entries.append({
                "year": p["year"],
                "page": p["page"],
                "image": cv_img,
                "source_pdf": p["source_pdf"],
            })

        # Filter entries without years (cover pages, etc.)
        entries_with_years = [e for e in entries if e["year"] is not None]
        entries_with_years.sort(key=lambda e: e["year"])

        n_images = len(entries_with_years)
        # Steps: n_images (object detect) + (n_images-1) (change detect) + n_images*3 (table gen)
        job.total_steps = n_images + max(0, n_images - 1) + n_images * 3

        # --- Step 2: CV Analysis ---
        job.status = JobStatus.ANALYZING

        # Detect boundaries
        boundary_type = "topo" if doc_type == "topo" else "aerial"
        for entry in entries_with_years:
            entry["boundary"] = detect_boundary(entry["image"], boundary_type)

        # Object detection
        image_results = []
        for i, entry in enumerate(entries_with_years):
            job.current_step = f"Analyzing {entry['year']} ({i+1}/{n_images})..."
            job.progress = i

            detections = await loop.run_in_executor(
                _executor, detect_objects, entry["image"]
            )

            boundary = entry.get("boundary")
            for det in detections:
                det["zone"] = classify_zone(det["bbox"], boundary, entry["image"].shape)

            image_results.append({
                "filename": f"{entry['source_pdf']}_p{entry['page']}",
                "year": entry["year"],
                "detections": detections,
            })

        # Change detection (pairwise)
        change_results = []
        for i in range(len(entries_with_years) - 1):
            a = entries_with_years[i]
            b = entries_with_years[i + 1]
            job.current_step = f"Comparing {a['year']} → {b['year']}..."
            job.progress = n_images + i

            result = await loop.run_in_executor(
                _executor, detect_changes,
                a["image"], b["image"], a["year"], b["year"],
            )

            boundary = b.get("boundary")
            for ch in result.get("changes", []):
                ch["zone"] = classify_zone(ch["bbox"], boundary, b["image"].shape)

            change_results.append(result)

        # --- Step 3: Generate tables via LLM ---
        job.status = JobStatus.GENERATING_TABLES
        job.current_step = "Generating ESA tables via local LLM..."
        base_progress = n_images + max(0, n_images - 1)
        job.progress = base_progress

        tables = await loop.run_in_executor(
            _executor, build_tables,
            image_results, change_results, llm_model, False,
        )

        job.tables_html = format_tables_html(tables)
        job.tables_text = format_tables_text(tables)

        # Also generate the JSON/MD report
        report_dir = REPORT_DIR / job.job_id
        report = generate_report(
            "uploaded", str(report_dir), image_results, change_results
        )
        job.report_json = {
            "summary": report.get("summary", {}),
            "rec_candidates": report.get("rec_candidates", []),
        }

        job.status = JobStatus.COMPLETE
        job.progress = job.total_steps
        job.current_step = "Complete"

    except Exception as e:
        job.status = JobStatus.ERROR
        job.error = str(e)
        import traceback
        traceback.print_exc()


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "progress": job.progress,
        "total_steps": job.total_steps,
        "current_step": job.current_step,
        "error": job.error,
    }


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status == JobStatus.ERROR:
        raise HTTPException(500, job.error)
    if job.status != JobStatus.COMPLETE:
        raise HTTPException(202, "Still processing")
    return {
        "tables_html": job.tables_html,
        "tables_text": job.tables_text,
        "rec_candidates": job.report_json.get("rec_candidates", []),
        "summary": job.report_json.get("summary", {}),
    }


@app.get("/api/results/{job_id}/html")
async def export_html(job_id: str):
    job = JOBS.get(job_id)
    if not job or job.status != JobStatus.COMPLETE:
        raise HTTPException(404)
    return HTMLResponse(content=f"""<!DOCTYPE html>
<html><head><title>ESA Historical Documentation Tables</title>
<style>
body{{font-family:Arial,sans-serif;max-width:900px;margin:40px auto;padding:0 20px}}
table{{width:100%;border-collapse:collapse;margin-bottom:20px}}
th,td{{border:1px solid #999;padding:8px;text-align:left}}
th{{background:#e8e8e8;font-size:0.9em}}
</style></head>
<body><h1>Historical Documentation Review</h1>{job.tables_html}</body></html>""")
