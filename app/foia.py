"""
FOIA / Records Request Automation — Phase I ESA

Automates the tedious process of requesting environmental records from
local, state, and federal agencies for Phase I ESAs.

Architecture:
  1. Agency Registry — database of agencies by jurisdiction (county/state)
     with request method (email, form URL, portal), contact info, and templates
  2. Request Generator — creates request letters/emails from site data
  3. Tracker — monitors sent requests, schedules follow-ups at 2 weeks
  4. Records of Communication — generates the ROC table for the ESA report

Agency request methods:
  - EMAIL: auto-sends via SMTP
  - FORM: opens pre-filled URL or generates fillable PDF
  - PORTAL: provides link + instructions
  - MAIL: generates printable letter
"""

import json
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data"
REGISTRY_FILE = DATA_DIR / "agency_registry.json"
REQUESTS_FILE = DATA_DIR / "foia_requests.json"


# ── Enums ────────────────────────────────────────────────────────────

class RequestMethod(str, Enum):
    EMAIL = "email"
    FORM = "form"
    PORTAL = "portal"
    MAIL = "mail"


class RequestStatus(str, Enum):
    DRAFT = "draft"
    SENT = "sent"
    FOLLOW_UP_DUE = "follow_up_due"
    FOLLOW_UP_SENT = "follow_up_sent"
    RECEIVED = "received"
    NO_RECORDS = "no_records"
    NO_RESPONSE = "no_response"


class AgencyType(str, Enum):
    FIRE = "fire_department"
    HEALTH = "health_department"
    BUILDING = "building_department"
    STATE_ENV = "state_environmental"
    WATER_BOARD = "water_board"
    PUBLIC_WORKS = "public_works"
    AIR_QUALITY = "air_quality"
    COUNTY_ENV = "county_environmental"


# ── Agency Registry ──────────────────────────────────────────────────

# Standard agencies needed per ASTM E1527-21
STANDARD_AGENCY_TYPES = {
    AgencyType.FIRE: {
        "records_requested": [
            "Underground storage tank (UST) records",
            "Hazardous materials business plans/permits",
            "Fire incident/spill reports",
        ],
        "info_needed": ["property_address", "parcel_number"],
    },
    AgencyType.HEALTH: {
        "records_requested": [
            "Water well permits and records",
            "Septic system permits",
            "Waste disposal permits",
            "Food facility permits (for environmental context)",
        ],
        "info_needed": ["property_address", "dates_of_interest"],
    },
    AgencyType.BUILDING: {
        "records_requested": [
            "Building permits (20+ year history)",
            "Certificate of occupancy records",
            "Demolition permits",
            "Tenant improvement permits",
        ],
        "info_needed": ["property_address", "historical_timeframe"],
    },
    AgencyType.STATE_ENV: {
        "records_requested": [
            "Cleanup site records",
            "Voluntary cleanup program records",
            "Hazardous waste generator records",
            "Enforcement actions",
        ],
        "info_needed": ["property_address", "business_names"],
    },
    AgencyType.WATER_BOARD: {
        "records_requested": [
            "Water quality violations",
            "Discharge permits (NPDES)",
            "Cleanup orders",
            "Site investigation reports",
        ],
        "info_needed": ["property_address", "business_names"],
    },
    AgencyType.PUBLIC_WORKS: {
        "records_requested": [
            "Waste management records",
            "Sewer permits",
            "Industrial waste discharge permits",
        ],
        "info_needed": ["property_address", "parcel_number"],
    },
}


def load_registry() -> dict:
    """Load the agency registry."""
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE) as f:
            return json.load(f)
    return {"agencies": {}, "jurisdictions": {}}


def save_registry(data: dict):
    DATA_DIR.mkdir(exist_ok=True)
    with open(REGISTRY_FILE, "w") as f:
        json.dump(data, f, indent=2)


def add_agency(
    jurisdiction: str,
    agency_type: str,
    name: str,
    method: str,
    contact_email: str = None,
    contact_phone: str = None,
    form_url: str = None,
    portal_url: str = None,
    mailing_address: str = None,
    notes: str = None,
    turnaround_days: int = 14,
) -> str:
    """
    Add an agency to the registry. Jurisdiction is typically
    'county_state' format, e.g., 'san_diego_ca' or 'orange_ca'.
    
    As we do more ESAs in a jurisdiction, the registry fills up
    and future requests in that area are instant.
    """
    reg = load_registry()
    
    agency_id = f"{jurisdiction}_{agency_type}"
    reg["agencies"][agency_id] = {
        "jurisdiction": jurisdiction,
        "agency_type": agency_type,
        "name": name,
        "method": method,
        "contact_email": contact_email,
        "contact_phone": contact_phone,
        "form_url": form_url,
        "portal_url": portal_url,
        "mailing_address": mailing_address,
        "notes": notes,
        "turnaround_days": turnaround_days,
        "last_used": None,
        "times_used": 0,
        "avg_response_days": None,
    }
    
    # Index by jurisdiction
    if jurisdiction not in reg["jurisdictions"]:
        reg["jurisdictions"][jurisdiction] = []
    if agency_id not in reg["jurisdictions"][jurisdiction]:
        reg["jurisdictions"][jurisdiction].append(agency_id)
    
    save_registry(reg)
    return agency_id


def get_agencies_for_jurisdiction(jurisdiction: str) -> list[dict]:
    """Get all registered agencies for a jurisdiction."""
    reg = load_registry()
    agency_ids = reg.get("jurisdictions", {}).get(jurisdiction, [])
    return [reg["agencies"][aid] for aid in agency_ids if aid in reg["agencies"]]


def find_jurisdiction(county: str, state: str) -> str:
    """Normalize jurisdiction key from county + state."""
    return f"{county.lower().replace(' ', '_')}_{state.lower()}"


# ── Request Generation ───────────────────────────────────────────────

EMAIL_TEMPLATE = """Subject: Environmental Records Request — {address}

Dear {agency_name} Records Division,

I am writing to request environmental records for the following property in connection with a Phase I Environmental Site Assessment (ESA) being conducted in general accordance with ASTM Standard Practice E 1527-21.

PROPERTY INFORMATION:
  Address: {address}
  City: {city}, {state} {zip_code}
  APN/Parcel Number: {parcel_number}
  {business_names_section}

RECORDS REQUESTED:
{records_list}

TIMEFRAME: Records from {start_year} to present.

Please provide any available records related to the above property. If no records are found, a written statement indicating "no records found" would be appreciated for our files.

If there is a fee associated with this request, please contact me with the amount prior to processing.

Thank you for your assistance. Please feel free to contact me with any questions.

Sincerely,
{consultant_name}
{consultant_title}
{company_name}
{consultant_email}
{consultant_phone}
Project Number: {project_number}
"""

MAIL_TEMPLATE = """{company_name}
{company_address}

{date}

{agency_name}
{agency_address}

RE: Environmental Records Request
    Property Address: {address}
    APN: {parcel_number}

Dear Records Division:

{body}

Sincerely,

{consultant_name}
{consultant_title}
{company_name}
"""


def generate_request(
    agency: dict,
    site_info: dict,
    consultant_info: dict,
) -> dict:
    """
    Generate a records request for a specific agency.
    
    site_info: {address, city, state, zip_code, parcel_number, 
                business_names, start_year, project_number}
    consultant_info: {name, title, company, email, phone, company_address}
    """
    agency_type = agency["agency_type"]
    standard = STANDARD_AGENCY_TYPES.get(agency_type, {})
    records = standard.get("records_requested", ["General environmental records"])
    records_list = "\n".join(f"  • {r}" for r in records)
    
    business_names = site_info.get("business_names", [])
    business_section = ""
    if business_names:
        business_section = "Known Business Names: " + ", ".join(business_names)
    
    request_id = str(uuid.uuid4())[:8]
    method = agency.get("method", "email")
    
    result = {
        "request_id": request_id,
        "agency_id": f"{agency['jurisdiction']}_{agency['agency_type']}",
        "agency_name": agency["name"],
        "agency_type": agency_type,
        "method": method,
        "status": RequestStatus.DRAFT.value,
        "created_at": datetime.now().isoformat(),
        "sent_at": None,
        "follow_up_due": None,
        "response_date": None,
        "response_summary": None,
        "site_address": site_info.get("address", ""),
        "project_number": site_info.get("project_number", ""),
    }
    
    if method == RequestMethod.EMAIL.value:
        result["email_to"] = agency.get("contact_email", "")
        result["email_subject"] = f"Environmental Records Request — {site_info.get('address', '')}"
        result["email_body"] = EMAIL_TEMPLATE.format(
            address=site_info.get("address", ""),
            city=site_info.get("city", ""),
            state=site_info.get("state", ""),
            zip_code=site_info.get("zip_code", ""),
            parcel_number=site_info.get("parcel_number", ""),
            business_names_section=business_section,
            records_list=records_list,
            start_year=site_info.get("start_year", "2000"),
            agency_name=agency["name"],
            consultant_name=consultant_info.get("name", ""),
            consultant_title=consultant_info.get("title", ""),
            company_name=consultant_info.get("company", ""),
            consultant_email=consultant_info.get("email", ""),
            consultant_phone=consultant_info.get("phone", ""),
            project_number=site_info.get("project_number", ""),
        )
    elif method == RequestMethod.FORM.value:
        result["form_url"] = agency.get("form_url", "")
        result["form_fields"] = {
            "property_address": site_info.get("address", ""),
            "parcel_number": site_info.get("parcel_number", ""),
            "city": site_info.get("city", ""),
            "state": site_info.get("state", ""),
            "zip_code": site_info.get("zip_code", ""),
            "requestor_name": consultant_info.get("name", ""),
            "requestor_email": consultant_info.get("email", ""),
            "requestor_phone": consultant_info.get("phone", ""),
            "company": consultant_info.get("company", ""),
            "records_requested": records_list,
        }
        result["instructions"] = f"Fill out form at: {agency.get('form_url', 'URL not set')}"
    elif method == RequestMethod.PORTAL.value:
        result["portal_url"] = agency.get("portal_url", "")
        result["instructions"] = (
            f"Submit request via portal: {agency.get('portal_url', '')}\n"
            f"Search for: {site_info.get('address', '')}\n"
            f"Notes: {agency.get('notes', '')}"
        )
    elif method == RequestMethod.MAIL.value:
        result["mailing_address"] = agency.get("mailing_address", "")
        result["letter_body"] = MAIL_TEMPLATE.format(
            company_name=consultant_info.get("company", ""),
            company_address=consultant_info.get("company_address", ""),
            date=datetime.now().strftime("%B %d, %Y"),
            agency_name=agency["name"],
            agency_address=agency.get("mailing_address", ""),
            address=site_info.get("address", ""),
            parcel_number=site_info.get("parcel_number", ""),
            body=EMAIL_TEMPLATE.format(
                address=site_info.get("address", ""),
                city=site_info.get("city", ""),
                state=site_info.get("state", ""),
                zip_code=site_info.get("zip_code", ""),
                parcel_number=site_info.get("parcel_number", ""),
                business_names_section=business_section,
                records_list=records_list,
                start_year=site_info.get("start_year", "2000"),
                agency_name=agency["name"],
                consultant_name=consultant_info.get("name", ""),
                consultant_title=consultant_info.get("title", ""),
                company_name=consultant_info.get("company", ""),
                consultant_email=consultant_info.get("email", ""),
                consultant_phone=consultant_info.get("phone", ""),
                project_number=site_info.get("project_number", ""),
            ),
            consultant_name=consultant_info.get("name", ""),
            consultant_title=consultant_info.get("title", ""),
        )
    
    return result


def generate_all_requests(
    jurisdiction: str,
    site_info: dict,
    consultant_info: dict,
) -> list[dict]:
    """
    Generate records requests for ALL registered agencies in a jurisdiction.
    Returns list of request dicts ready to send/submit.
    """
    agencies = get_agencies_for_jurisdiction(jurisdiction)
    
    if not agencies:
        # Return a list of what's needed so user knows what to register
        needed = []
        for atype, info in STANDARD_AGENCY_TYPES.items():
            needed.append({
                "agency_type": atype.value,
                "records_requested": info["records_requested"],
                "status": "no_agency_registered",
                "message": f"No {atype.value} registered for {jurisdiction}. Add one via the agency registry.",
            })
        return needed
    
    requests = []
    for agency in agencies:
        req = generate_request(agency, site_info, consultant_info)
        requests.append(req)
    
    return requests


# ── Request Tracking ─────────────────────────────────────────────────

def load_requests() -> dict:
    """Load all tracked requests."""
    if REQUESTS_FILE.exists():
        with open(REQUESTS_FILE) as f:
            return json.load(f)
    return {"requests": {}}


def save_requests(data: dict):
    DATA_DIR.mkdir(exist_ok=True)
    with open(REQUESTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def track_request(request: dict) -> str:
    """Save a generated request to the tracker."""
    data = load_requests()
    rid = request["request_id"]
    data["requests"][rid] = request
    save_requests(data)
    return rid


def mark_sent(request_id: str, follow_up_days: int = 14):
    """Mark a request as sent and schedule follow-up."""
    data = load_requests()
    req = data["requests"].get(request_id)
    if req:
        req["status"] = RequestStatus.SENT.value
        req["sent_at"] = datetime.now().isoformat()
        req["follow_up_due"] = (datetime.now() + timedelta(days=follow_up_days)).isoformat()
        save_requests(data)


def mark_received(request_id: str, summary: str = ""):
    """Mark a request as received with response summary."""
    data = load_requests()
    req = data["requests"].get(request_id)
    if req:
        req["status"] = RequestStatus.RECEIVED.value
        req["response_date"] = datetime.now().isoformat()
        req["response_summary"] = summary
        
        # Update agency registry with response time
        _update_agency_stats(req)
        save_requests(data)


def mark_no_records(request_id: str):
    """Mark as responded with no records found."""
    data = load_requests()
    req = data["requests"].get(request_id)
    if req:
        req["status"] = RequestStatus.NO_RECORDS.value
        req["response_date"] = datetime.now().isoformat()
        req["response_summary"] = "No records found"
        _update_agency_stats(req)
        save_requests(data)


def get_follow_ups_due() -> list[dict]:
    """Get all requests where follow-up is due."""
    data = load_requests()
    due = []
    now = datetime.now()
    for req in data["requests"].values():
        if req["status"] == RequestStatus.SENT.value and req.get("follow_up_due"):
            follow_up = datetime.fromisoformat(req["follow_up_due"])
            if now >= follow_up:
                req["days_waiting"] = (now - datetime.fromisoformat(req["sent_at"])).days
                due.append(req)
    return due


def get_requests_by_project(project_number: str) -> list[dict]:
    """Get all requests for a specific project."""
    data = load_requests()
    return [r for r in data["requests"].values() if r.get("project_number") == project_number]


def _update_agency_stats(req: dict):
    """Update agency registry with response time stats."""
    reg = load_registry()
    agency_id = req.get("agency_id", "")
    agency = reg.get("agencies", {}).get(agency_id)
    if agency and req.get("sent_at") and req.get("response_date"):
        sent = datetime.fromisoformat(req["sent_at"])
        received = datetime.fromisoformat(req["response_date"])
        days = (received - sent).days
        
        agency["last_used"] = datetime.now().isoformat()
        agency["times_used"] = agency.get("times_used", 0) + 1
        
        # Running average
        prev_avg = agency.get("avg_response_days")
        n = agency["times_used"]
        if prev_avg and n > 1:
            agency["avg_response_days"] = round((prev_avg * (n - 1) + days) / n, 1)
        else:
            agency["avg_response_days"] = days
        
        save_registry(reg)


# ── Records of Communication (ROC) Table ─────────────────────────────

def generate_roc_table(project_number: str) -> list[dict]:
    """
    Generate the Records of Communication table for the ESA report.
    This is the table that goes in Section 4 / Appendix showing
    all agency contacts and their responses.
    """
    requests = get_requests_by_project(project_number)
    
    roc_rows = []
    for req in sorted(requests, key=lambda r: r.get("sent_at") or ""):
        status = req.get("status", "draft")
        
        if status == RequestStatus.RECEIVED.value:
            response_text = req.get("response_summary", "Records received")
        elif status == RequestStatus.NO_RECORDS.value:
            response_text = "No records found"
        elif status == RequestStatus.NO_RESPONSE.value:
            response_text = "No response received as of report date"
        elif status == RequestStatus.SENT.value:
            response_text = "Response pending"
        else:
            response_text = "Not yet submitted"
        
        roc_rows.append({
            "agency": req.get("agency_name", ""),
            "agency_type": req.get("agency_type", ""),
            "date_requested": req.get("sent_at", "")[:10] if req.get("sent_at") else "—",
            "date_received": req.get("response_date", "")[:10] if req.get("response_date") else "—",
            "method": req.get("method", ""),
            "response": response_text,
        })
    
    return roc_rows


def generate_roc_html(project_number: str) -> str:
    """Generate HTML table for the ROC."""
    rows = generate_roc_table(project_number)
    
    html = [
        '<h2>Records of Communication</h2>',
        '<table border="1" cellpadding="8" cellspacing="0" style="border-collapse:collapse;width:100%">',
        '<thead><tr>',
        '<th>Agency</th><th>Type</th><th>Method</th>',
        '<th>Date Requested</th><th>Date Received</th><th>Response</th>',
        '</tr></thead><tbody>',
    ]
    
    for row in rows:
        html.append(
            f"<tr>"
            f"<td>{row['agency']}</td>"
            f"<td>{row['agency_type']}</td>"
            f"<td>{row['method']}</td>"
            f"<td>{row['date_requested']}</td>"
            f"<td>{row['date_received']}</td>"
            f"<td>{row['response']}</td>"
            f"</tr>"
        )
    
    html.append("</tbody></table>")
    return "\n".join(html)
