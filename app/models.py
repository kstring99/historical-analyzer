from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import uuid
import time


class DocType(str, Enum):
    AERIAL = "aerial"
    TOPO = "topo"
    CITY_DIRECTORY = "city_directory"
    UNKNOWN = "unknown"


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"


class Zone(str, Enum):
    SUBJECT = "subject"
    ADJOINING = "adjoining"
    SURROUNDING = "surrounding"


@dataclass
class TableRow:
    """Single row in the output table."""
    year_range: str  # e.g., "1946 - 1978"
    issues_noted: str  # "Yes" or "No"
    observations: str  # Description text or occupant listing


@dataclass
class ZoneTable:
    """One table (Subject, Adjoining, or Surrounding)."""
    zone: Zone
    rows: list[TableRow] = field(default_factory=list)


@dataclass
class PageResult:
    """Raw analysis result for a single PDF page."""
    page_number: int
    year: str
    subject_obs: str = ""
    adjoining_obs: str = ""
    surrounding_obs: str = ""
    issues_noted: dict = field(default_factory=lambda: {
        "subject": "No", "adjoining": "No", "surrounding": "No"
    })
    raw_response: str = ""


@dataclass
class DocumentResult:
    doc_type: DocType
    filename: str
    metadata: dict = field(default_factory=dict)
    years_reviewed: list[str] = field(default_factory=list)
    tables: list[ZoneTable] = field(default_factory=list)  # 3 tables per doc
    pages: list[PageResult] = field(default_factory=list)  # raw page results


@dataclass
class Job:
    job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    documents: list[DocumentResult] = field(default_factory=list)
    summary: str = ""  # Cross-referencing summary paragraph
    progress: int = 0
    total_pages: int = 0
    current_file: str = ""
    error: Optional[str] = None
    # User-provided addresses for city directory matching
    subject_address: str = ""
    adjoining_addresses: list[str] = field(default_factory=list)
