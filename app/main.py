import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from app.models import Job, JobStatus, Zone
from app.llm import get_provider
from app.processor import process_document, generate_summary
from app import foia
from app import site_data

app = FastAPI(title="Historical Documentation Analyzer")

JOBS: dict[str, Job] = {}
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/upload")
async def upload_pdfs(
    files: list[UploadFile] = File(...),
    provider: str = Form(default="anthropic"),
    api_key: str = Form(default=""),
    subject_address: str = Form(default=""),
    adjoining_addresses: str = Form(default=""),
):
    """Upload PDF(s) and start processing."""
    job = Job()
    job.subject_address = subject_address
    job.adjoining_addresses = [a.strip() for a in adjoining_addresses.split("\n") if a.strip()]
    JOBS[job.job_id] = job

    job_dir = UPLOAD_DIR / job.job_id
    job_dir.mkdir(exist_ok=True)

    saved_files = []
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(400, f"Only PDF files accepted, got: {f.filename}")
        dest = job_dir / f.filename
        with open(dest, "wb") as out:
            content = await f.read()
            out.write(content)
        saved_files.append(str(dest))

    asyncio.create_task(_process_job(job, saved_files, provider, api_key or None))
    return {"job_id": job.job_id, "files": [f.filename for f in files]}


async def _process_job(job: Job, pdf_paths: list[str], provider: str, api_key: str | None):
    try:
        job.status = JobStatus.PROCESSING
        llm = get_provider(provider, api_key)

        from pdf2image import pdfinfo_from_path
        total = sum(pdfinfo_from_path(p)["Pages"] for p in pdf_paths)
        job.total_pages = total
        processed = 0

        for path in pdf_paths:
            filename = Path(path).name
            job.current_file = filename

            async def progress_cb(msg: str, current: int, page_total: int):
                nonlocal processed
                job.progress = processed + current
                job.current_file = msg

            result = await process_document(
                path, llm, progress_cb,
                subject_address=job.subject_address,
                adjoining_addresses=job.adjoining_addresses,
            )
            job.documents.append(result)

            info = pdfinfo_from_path(path)
            processed += info["Pages"]
            job.progress = processed

        # Generate cross-referencing summary
        if len(job.documents) > 0:
            job.current_file = "Generating cross-reference summary..."
            job.summary = await generate_summary(job.documents, llm)

        job.status = JobStatus.COMPLETE
        job.progress = job.total_pages
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
        raise HTTPException(500, job.error)
    if job.status != JobStatus.COMPLETE:
        raise HTTPException(202, "Still processing")

    results = []
    for doc in job.documents:
        doc_result = {
            "doc_type": doc.doc_type.value,
            "filename": doc.filename,
            "metadata": doc.metadata,
            "years_reviewed": doc.years_reviewed,
            "tables": [],
        }
        for table in doc.tables:
            doc_result["tables"].append({
                "zone": table.zone.value,
                "zone_label": {
                    "subject": "Subject Property",
                    "adjoining": "Adjoining Properties",
                    "surrounding": "Surrounding Properties",
                }[table.zone.value],
                "rows": [
                    {
                        "year_range": r.year_range,
                        "issues_noted": r.issues_noted,
                        "observations": r.observations,
                    }
                    for r in table.rows
                ],
            })
        results.append(doc_result)
    return {"job_id": job.job_id, "results": results, "summary": job.summary}


# ── FOIA Pages & API Routes ──────────────────────────────────────────

@app.get("/foia")
async def foia_page():
    return FileResponse(str(STATIC_DIR / "foia.html"))


@app.get("/api/foia/jurisdictions")
async def get_jurisdictions():
    """Return all known jurisdictions from the registry."""
    reg = foia.load_registry()
    jurisdictions = list(reg.get("jurisdictions", {}).keys())
    return {"jurisdictions": jurisdictions}


# ── FOIA / Records Request Routes ────────────────────────────────────

@app.get("/api/foia/agencies/{jurisdiction}")
async def get_agencies(jurisdiction: str):
    """Get all registered agencies for a jurisdiction."""
    agencies = foia.get_agencies_for_jurisdiction(jurisdiction)
    if not agencies:
        return {"jurisdiction": jurisdiction, "agencies": [], "needed": list(foia.STANDARD_AGENCY_TYPES.keys())}
    return {"jurisdiction": jurisdiction, "agencies": agencies}


@app.post("/api/foia/agencies")
async def add_agency(
    jurisdiction: str = Form(...),
    agency_type: str = Form(...),
    name: str = Form(...),
    method: str = Form(...),
    contact_email: str = Form(default=""),
    contact_phone: str = Form(default=""),
    form_url: str = Form(default=""),
    portal_url: str = Form(default=""),
    mailing_address: str = Form(default=""),
    notes: str = Form(default=""),
):
    """Register a new agency in the registry."""
    agency_id = foia.add_agency(
        jurisdiction=jurisdiction,
        agency_type=agency_type,
        name=name,
        method=method,
        contact_email=contact_email or None,
        contact_phone=contact_phone or None,
        form_url=form_url or None,
        portal_url=portal_url or None,
        mailing_address=mailing_address or None,
        notes=notes or None,
    )
    return {"agency_id": agency_id, "status": "registered"}


@app.post("/api/foia/generate")
async def generate_requests(
    jurisdiction: str = Form(...),
    address: str = Form(...),
    city: str = Form(...),
    state: str = Form(...),
    zip_code: str = Form(...),
    parcel_number: str = Form(default=""),
    business_names: str = Form(default=""),
    start_year: str = Form(default="2000"),
    project_number: str = Form(default=""),
    consultant_name: str = Form(default=""),
    consultant_title: str = Form(default=""),
    consultant_email: str = Form(default=""),
    consultant_phone: str = Form(default=""),
    company_name: str = Form(default=""),
    company_address: str = Form(default=""),
):
    """Generate records requests for all agencies in a jurisdiction."""
    site_info = {
        "address": address,
        "city": city,
        "state": state,
        "zip_code": zip_code,
        "parcel_number": parcel_number,
        "business_names": [b.strip() for b in business_names.split("\n") if b.strip()],
        "start_year": start_year,
        "project_number": project_number,
    }
    consultant_info = {
        "name": consultant_name,
        "title": consultant_title,
        "email": consultant_email,
        "phone": consultant_phone,
        "company": company_name,
        "company_address": company_address,
    }
    
    requests = foia.generate_all_requests(jurisdiction, site_info, consultant_info)
    
    # Track all generated requests
    for req in requests:
        if req.get("request_id"):
            foia.track_request(req)
    
    return {"requests": requests, "count": len(requests)}


@app.post("/api/foia/requests/{request_id}/sent")
async def mark_request_sent(request_id: str):
    """Mark a request as sent."""
    foia.mark_sent(request_id)
    return {"status": "sent", "follow_up_due": (datetime.now() + timedelta(days=14)).isoformat()[:10]}


@app.post("/api/foia/requests/{request_id}/received")
async def mark_request_received(request_id: str, summary: str = Form(default="")):
    """Mark a request as received with optional summary."""
    foia.mark_received(request_id, summary)
    return {"status": "received"}


@app.post("/api/foia/requests/{request_id}/no-records")
async def mark_no_records(request_id: str):
    """Mark a request as no records found."""
    foia.mark_no_records(request_id)
    return {"status": "no_records"}


@app.get("/api/foia/follow-ups")
async def get_follow_ups():
    """Get all requests where follow-up is due."""
    due = foia.get_follow_ups_due()
    return {"follow_ups": due, "count": len(due)}


@app.get("/api/foia/project/{project_number}")
async def get_project_requests(project_number: str):
    """Get all requests for a project."""
    requests = foia.get_requests_by_project(project_number)
    return {"requests": requests, "count": len(requests)}


@app.get("/api/foia/roc/{project_number}")
async def get_roc(project_number: str):
    """Generate Records of Communication table."""
    requests = foia.get_requests_by_project(project_number)
    if not requests:
        return HTMLResponse(
            content='<div style="text-align:center;padding:40px;color:#718096;">'
            '<p style="font-size:1.2em;margin-bottom:8px;">No requests found for this project.</p>'
            '<p>Generate requests first using the <strong>New Project Request</strong> section above, '
            'then come back here to export the ROC table.</p></div>'
        )
    html = foia.generate_roc_html(project_number)
    return HTMLResponse(content=html)


# ── Site Data Routes (Government Data Puller) ────────────────────────

@app.post("/api/site-data/pull")
async def pull_site_data(address: str = Form(...)):
    """Pull all government site data for an address (sections 3.5.1-3.5.4)."""
    result = await site_data.pull_all_site_data(address)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.get("/site-data")
async def site_data_page():
    return FileResponse(str(STATIC_DIR / "site_data.html"))


# ── Export Routes ────────────────────────────────────────────────────

@app.post("/api/export/{job_id}")
async def export_results(job_id: str, format: str = Form(default="html")):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.COMPLETE:
        raise HTTPException(400, "Job not complete")

    html_parts = []
    
    # Summary paragraph first
    if job.summary:
        html_parts.append(f"<h2>Historical Documentation Summary</h2>")
        html_parts.append(f"<p>{job.summary}</p>")
    
    for doc in job.documents:
        section_title = {
            "aerial": "Aerial Photographs",
            "topo": "Topographic Maps",
            "city_directory": "Street Directories",
        }.get(doc.doc_type.value, doc.doc_type.value)

        table_title = {
            "aerial": "AERIAL PHOTOGRAPH SUMMARY",
            "topo": "TOPOGRAPHIC MAP SUMMARY",
            "city_directory": "STREET DIRECTORY SUMMARY",
        }.get(doc.doc_type.value, "SUMMARY")

        obs_col = "Occupants" if doc.doc_type.value == "city_directory" else "Observations"

        html_parts.append(f"<h2>{section_title}</h2>")
        if doc.years_reviewed:
            years_str = ", ".join(doc.years_reviewed)
            html_parts.append(f"<p>Years reviewed: {years_str}</p>")

        for table in doc.tables:
            zone_label = {
                "subject": "Subject Property",
                "adjoining": "Adjoining Properties",
                "surrounding": "Surrounding Properties",
            }[table.zone.value]

            html_parts.append(f"<h3>{zone_label}</h3>")
            html_parts.append(f'<p style="text-align:center;font-weight:bold;margin-bottom:4px">{table_title} - {zone_label}</p>')
            html_parts.append('<table border="1" cellpadding="8" cellspacing="0" style="border-collapse:collapse;width:100%">')
            html_parts.append(f"<thead><tr><th>Year</th><th>Issues Noted</th><th>{obs_col}</th></tr></thead><tbody>")
            for row in table.rows:
                html_parts.append(
                    f"<tr>"
                    f"<td style='white-space:nowrap;vertical-align:top'>{row.year_range}</td>"
                    f"<td style='text-align:center;vertical-align:top'>{row.issues_noted}</td>"
                    f"<td>{row.observations}</td>"
                    f"</tr>"
                )
            html_parts.append("</tbody></table><br>")

    full_html = "\n".join(html_parts)
    return HTMLResponse(content=f"""<!DOCTYPE html>
<html><head><title>ESA Historical Documentation Review</title>
<style>
body{{font-family:Arial,sans-serif;max-width:900px;margin:40px auto;padding:0 20px}}
table{{width:100%;border-collapse:collapse;margin-bottom:20px}}
th,td{{border:1px solid #999;padding:8px;text-align:left}}
th{{background:#e8e8e8;font-size:0.9em}}
h2{{color:#1a365d;margin-top:30px;border-bottom:2px solid #1a365d;padding-bottom:8px}}
h3{{color:#333;margin-top:20px}}
</style></head>
<body><h1>Historical Documentation Review</h1>{full_html}</body></html>""")
