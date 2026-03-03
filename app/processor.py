import io
import asyncio
import concurrent.futures
import re
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image

from app.llm import LLMProvider
from app.models import (
    DocType, PageResult, DocumentResult, ZoneTable, Zone, TableRow
)

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


# --- Document type classification ---

def classify_document(filename: str, cover_text: str = "") -> DocType:
    name = filename.lower()
    if "aerial" in name:
        return DocType.AERIAL
    if "topo" in name:
        return DocType.TOPO
    if "cd" in name or "city" in name or "director" in name:
        return DocType.CITY_DIRECTORY
    cover = cover_text.lower()
    if "aerial" in cover:
        return DocType.AERIAL
    if "topographic" in cover or "topo" in cover:
        return DocType.TOPO
    if "city directory" in cover or "directory" in cover:
        return DocType.CITY_DIRECTORY
    return DocType.UNKNOWN


# --- PDF page extraction ---

def extract_pages(pdf_path: str, dpi: int = 200) -> list[Image.Image]:
    return convert_from_path(pdf_path, dpi=dpi)


def image_to_bytes(img: Image.Image, fmt: str = "JPEG", quality: int = 85) -> tuple[bytes, str]:
    """Convert PIL Image to bytes. Returns (bytes, media_type)."""
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    media_type = "image/jpeg" if fmt.upper() in ("JPEG", "JPG") else "image/png"
    return buf.getvalue(), media_type


# --- Prompts ---

COVER_PAGE_PROMPT = """This is the cover page of an ERIS historical documentation package for a Phase I ESA.
Extract metadata:
- Property Address / Location
- Coordinates (lat/long)
- Project Property name
- Project Number
- Order Number

Return as key: value list. Omit fields not found."""

AERIAL_PROMPT = """You are analyzing an aerial photograph for a Phase I Environmental Site Assessment.

A green rectangle marks the subject property boundary on this image.

Analyze THREE zones and provide observations for each:

1. SUBJECT PROPERTY — What is WITHIN the green rectangle? Describe land use, structures, vegetation, development.

2. ADJOINING PROPERTIES — What is immediately adjacent to the green rectangle (properties sharing a boundary)? Note directional references (to the north, south, east, west). Describe land use, structures.

3. SURROUNDING PROPERTIES — What is in the wider visible area beyond adjoining properties? General land use, development patterns, notable features.

4. ISSUES — ONLY mark "Yes" if you can VISUALLY IDENTIFY specific environmental concerns such as:
   - Fuel canopies, gas station structures, UST fill ports
   - Above-ground storage tanks (ASTs) or clusters of drums/containers
   - Soil staining, discolored ground, or distressed vegetation patterns
   - Industrial equipment, smokestacks, or processing facilities
   - Auto salvage/junkyard with visible vehicle accumulation
   - Landfill-like mounding or waste disposal evidence
   
   Regular buildings, parking lots, residential structures, commercial buildings, and normal development are NOT concerns. A machine shop or auto repair facility looks like any commercial building from the air — do NOT flag it as a concern unless you see specific visual evidence (staining, tanks, drums, etc.)
   
   If "Yes", state EXACTLY what you see (e.g., "Yes — fuel canopy and paved forecourt consistent with gasoline service station visible at adjoining property to the south").

Extract the YEAR from the footer bar.

Format your response EXACTLY as:
YEAR: [year from footer]
SUBJECT: [description starting with "The subject property appears to..."]
SUBJECT_ISSUES: [No, or Yes — with specific visual evidence described]
ADJOINING: [description of adjoining properties with directional references]
ADJOINING_ISSUES: [No, or Yes — with specific visual evidence described]
SURROUNDING: [description of surrounding area]
SURROUNDING_ISSUES: [No, or Yes — with specific visual evidence described]"""

TOPO_PROMPT_TEMPLATE = """You are analyzing a USGS topographic map for a Phase I Environmental Site Assessment.

The subject property is located at approximately {coordinates}. {property_context}

Analyze THREE zones:

1. SUBJECT PROPERTY — What is at the property location? Terrain, any structures shown, land use indicators.

2. ADJOINING PROPERTIES — What is immediately surrounding the property location? Roads, structures, water features, directional references.

3. SURROUNDING PROPERTIES — Wider area visible on the map. General terrain, development, notable features.

4. ISSUES — ONLY mark "Yes" if the topographic map shows specific environmental concerns:
   - Mine symbols, quarry symbols, or mining-related features (tailings, adits)
   - Landfill or disposal site symbols
   - Industrial facility symbols
   - Pipeline or tank farm symbols
   - Sewage treatment or water treatment plant symbols
   
   Normal development (roads, buildings, residential areas) is NOT a concern. If "Yes", state EXACTLY what map feature you see.

Extract the YEAR from the footer.

Format your response EXACTLY as:
YEAR: [year]
SUBJECT: [description starting with "The subject property appears to..."]
SUBJECT_ISSUES: [No, or Yes — with specific map evidence described]
ADJOINING: [description with directional references]
ADJOINING_ISSUES: [No, or Yes — with specific map evidence described]
SURROUNDING: [description]
SURROUNDING_ISSUES: [No, or Yes — with specific map evidence described]"""

CITY_DIR_PROMPT = """You are analyzing a city directory page for a Phase I Environmental Site Assessment.

Extract ALL directory entries on this page. For each entry, note:
- Year (from header)
- Street Name (from section header)
- Street Number
- Occupant/Business Name
- Use Classification
- Whether the use is environmentally concerning (gas station, dry cleaner, auto repair, chemical storage, manufacturing, printing, etc.)

Format your response as:
YEAR: [year]
STREET: [street name]
ENTRIES:
[number] | [occupant] | [classification] | [ENV_FLAG if concerning, else NONE]
..."""

CITY_DIR_SYNTHESIS_PROMPT = """You are writing the city directory section for a Phase I ESA.

Subject property address: {subject_address}
Adjoining property addresses: {adjoining_addresses}

Raw directory data:
{raw_entries}

For EACH year that has data, create entries for three zones:

1. SUBJECT PROPERTY — listings at the subject property address. If no listing exists for a year, note "No listing identified for the subject property address."

2. ADJOINING PROPERTIES — listings at adjoining property addresses. If no listings, note "No listings identified for adjoining property addresses."

3. SURROUNDING — any other listings in the area that could be environmentally relevant.

ISSUES NOTED RULES — ONLY mark "Yes" if a listing indicates an environmentally concerning use:
- Gasoline service station, filling station, fuel dealer
- Auto repair, auto body, auto salvage/wrecker
- Dry cleaner, laundry (using solvents)
- Chemical manufacturing, storage, or distribution
- Printing/lithography shop
- Machine shop, metalworking, plating
- Paint store, hardware store with chemical sales
- Pest control, exterminator
- Photo processing/darkroom

Regular residential, retail, restaurants, schools, offices, churches are NOT concerns. If "Yes", state the specific business name and use type (e.g., "Yes — Joe's Texaco, gasoline service station").

Format EXACTLY as:
YEAR: [year]
SUBJECT: [occupant listing or "No listing identified for the subject property address."]
SUBJECT_ISSUES: [No, or Yes — with specific business and use type]
ADJOINING: [occupant listings or "No listings identified for adjoining property addresses."]
ADJOINING_ISSUES: [No, or Yes — with specific business and use type]
SURROUNDING: [relevant nearby listings or "No environmentally significant listings identified in the surrounding area."]
SURROUNDING_ISSUES: [No, or Yes — with specific business and use type]"""


# --- Response parsing ---

def _parse_three_zone_response(response: str) -> PageResult:
    """Parse a structured three-zone response into PageResult."""
    result = PageResult(page_number=0, year="")
    
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("YEAR:"):
            result.year = line.split(":", 1)[1].strip()
        elif line.startswith("SUBJECT:") and not line.startswith("SUBJECT_ISSUES"):
            result.subject_obs = line.split(":", 1)[1].strip()
        elif line.startswith("SUBJECT_ISSUES:"):
            result.issues_noted["subject"] = line.split(":", 1)[1].strip()
        elif line.startswith("ADJOINING:") and not line.startswith("ADJOINING_ISSUES"):
            result.adjoining_obs = line.split(":", 1)[1].strip()
        elif line.startswith("ADJOINING_ISSUES:"):
            result.issues_noted["adjoining"] = line.split(":", 1)[1].strip()
        elif line.startswith("SURROUNDING:") and not line.startswith("SURROUNDING_ISSUES"):
            result.surrounding_obs = line.split(":", 1)[1].strip()
        elif line.startswith("SURROUNDING_ISSUES:"):
            result.issues_noted["surrounding"] = line.split(":", 1)[1].strip()
    
    return result


# --- Year range grouping ---

def _group_year_ranges(pages: list[PageResult], zone: str) -> list[TableRow]:
    """Group consecutive years with similar observations into year ranges."""
    if not pages:
        return []
    
    # Sort by year
    sorted_pages = sorted(pages, key=lambda p: p.year)
    
    # Get observation for the zone
    def get_obs(p: PageResult) -> str:
        if zone == "subject":
            return p.subject_obs
        elif zone == "adjoining":
            return p.adjoining_obs
        return p.surrounding_obs
    
    def get_issues(p: PageResult) -> str:
        return p.issues_noted.get(zone, "No")
    
    # Group consecutive pages with similar observations
    groups = []
    current_group = [sorted_pages[0]]
    
    for page in sorted_pages[1:]:
        prev_obs = get_obs(current_group[-1]).lower().strip()
        curr_obs = get_obs(page).lower().strip()
        
        # Simple similarity check — if observations are very similar, group them
        # In practice, the LLM should produce similar descriptions for similar years
        if _observations_similar(prev_obs, curr_obs) and get_issues(page) == get_issues(current_group[-1]):
            current_group.append(page)
        else:
            groups.append(current_group)
            current_group = [page]
    groups.append(current_group)
    
    # Convert groups to TableRows with year ranges
    rows = []
    for group in groups:
        if len(group) == 1:
            year_range = group[0].year
        else:
            year_range = f"{group[0].year} - {group[-1].year}"
        
        # Use the most detailed observation from the group (usually the last one)
        obs = get_obs(group[-1]) or get_obs(group[0])
        issues = get_issues(group[-1])
        
        rows.append(TableRow(
            year_range=year_range,
            issues_noted=issues,
            observations=obs
        ))
    
    return rows


def _observations_similar(obs1: str, obs2: str) -> bool:
    """Check if two observations are similar enough to group."""
    if not obs1 or not obs2:
        return False
    
    # Extract key phrases
    words1 = set(re.findall(r'\b\w{4,}\b', obs1.lower()))
    words2 = set(re.findall(r'\b\w{4,}\b', obs2.lower()))
    
    if not words1 or not words2:
        return False
    
    overlap = len(words1 & words2) / max(len(words1), len(words2))
    return overlap > 0.6  # 60% word overlap = similar enough to group


# --- Page analysis ---

def _analyze_cover(llm: LLMProvider, img: Image.Image) -> dict:
    img_bytes, media_type = image_to_bytes(img)
    response = llm.analyze_image(img_bytes, COVER_PAGE_PROMPT, media_type=media_type)
    metadata = {"raw": response}
    for line in response.split("\n"):
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip().lower()
            val = val.strip()
            if "address" in key or "location" in key:
                metadata["address"] = val
            elif "coordinate" in key or "lat" in key:
                metadata["coordinates"] = val
            elif "project" in key and "number" in key:
                metadata["project_number"] = val
            elif "project" in key and "property" in key:
                metadata["project_name"] = val
            elif "order" in key:
                metadata["order_number"] = val
    return metadata


def _analyze_aerial_page(llm: LLMProvider, img: Image.Image, page_num: int) -> PageResult:
    img_bytes, media_type = image_to_bytes(img)
    response = llm.analyze_image(img_bytes, AERIAL_PROMPT, media_type=media_type)
    result = _parse_three_zone_response(response)
    result.page_number = page_num
    result.raw_response = response
    return result


def _analyze_topo_page(llm: LLMProvider, img: Image.Image, page_num: int, metadata: dict) -> PageResult:
    coords = metadata.get("coordinates", "coordinates not available")
    context = ""
    if "address" in metadata:
        context += f"Located near {metadata['address']}."
    
    prompt = TOPO_PROMPT_TEMPLATE.format(coordinates=coords, property_context=context)
    img_bytes, media_type = image_to_bytes(img)
    response = llm.analyze_image(img_bytes, prompt, media_type=media_type)
    result = _parse_three_zone_response(response)
    result.page_number = page_num
    result.raw_response = response
    return result


def _analyze_cd_page(llm: LLMProvider, img: Image.Image, page_num: int) -> PageResult:
    img_bytes, media_type = image_to_bytes(img)
    response = llm.analyze_image(img_bytes, CITY_DIR_PROMPT, media_type=media_type)
    year = ""
    for line in response.split("\n"):
        if line.startswith("YEAR:"):
            year = line.split(":", 1)[1].strip()
            break
    return PageResult(page_number=page_num, year=year, raw_response=response)


# --- Build zone tables ---

def _build_zone_tables(pages: list[PageResult]) -> list[ZoneTable]:
    """Build three ZoneTables from analyzed pages."""
    tables = []
    for zone in [Zone.SUBJECT, Zone.ADJOINING, Zone.SURROUNDING]:
        rows = _group_year_ranges(pages, zone.value)
        tables.append(ZoneTable(zone=zone, rows=rows))
    return tables


# --- Main processing ---

async def process_document(
    pdf_path: str,
    llm: LLMProvider,
    progress_callback=None,
    subject_address: str = "",
    adjoining_addresses: list[str] = None,
) -> DocumentResult:
    """Process a complete PDF document."""
    filename = Path(pdf_path).name
    doc_type = classify_document(filename)
    
    pages = extract_pages(pdf_path)
    total = len(pages)
    
    if progress_callback:
        await progress_callback(f"Analyzing cover page of {filename}...", 0, total)
    
    loop = asyncio.get_event_loop()
    metadata = await loop.run_in_executor(_executor, _analyze_cover, llm, pages[0])
    
    if doc_type == DocType.UNKNOWN:
        doc_type = classify_document(filename, metadata.get("raw", ""))
    
    result = DocumentResult(doc_type=doc_type, filename=filename, metadata=metadata)
    content_pages = pages[1:]
    
    if doc_type == DocType.AERIAL:
        page_results = []
        for i, page_img in enumerate(content_pages):
            if progress_callback:
                await progress_callback(f"Analyzing aerial {i+1}/{len(content_pages)}...", i + 1, total)
            pr = await loop.run_in_executor(_executor, _analyze_aerial_page, llm, page_img, i + 2)
            page_results.append(pr)
        
        result.pages = page_results
        result.years_reviewed = sorted(set(p.year for p in page_results if p.year))
        result.tables = _build_zone_tables(page_results)
    
    elif doc_type == DocType.TOPO:
        page_results = []
        for i, page_img in enumerate(content_pages):
            if progress_callback:
                await progress_callback(f"Analyzing topo {i+1}/{len(content_pages)}...", i + 1, total)
            pr = await loop.run_in_executor(
                _executor, _analyze_topo_page, llm, page_img, i + 2, metadata
            )
            page_results.append(pr)
        
        result.pages = page_results
        result.years_reviewed = sorted(set(p.year for p in page_results if p.year))
        result.tables = _build_zone_tables(page_results)
    
    elif doc_type == DocType.CITY_DIRECTORY:
        # First pass: extract raw entries from each page
        raw_pages = []
        for i, page_img in enumerate(content_pages):
            if progress_callback:
                await progress_callback(f"Reading directory {i+1}/{len(content_pages)}...", i + 1, total)
            pr = await loop.run_in_executor(_executor, _analyze_cd_page, llm, page_img, i + 2)
            raw_pages.append(pr)
        
        # Second pass: synthesize with address matching
        if progress_callback:
            await progress_callback("Synthesizing directory entries...", total - 1, total)
        
        all_entries = "\n\n---\n\n".join(p.raw_response for p in raw_pages)
        adj_str = ", ".join(adjoining_addresses) if adjoining_addresses else "Not provided"
        
        synthesis_prompt = CITY_DIR_SYNTHESIS_PROMPT.format(
            subject_address=subject_address or "Not provided",
            adjoining_addresses=adj_str,
            raw_entries=all_entries
        )
        synthesis = await loop.run_in_executor(_executor, llm.generate_text, synthesis_prompt)
        
        # Parse synthesized three-zone results
        page_results = []
        current = PageResult(page_number=0, year="")
        
        for line in synthesis.split("\n"):
            line = line.strip()
            if line.startswith("YEAR:"):
                if current.year:
                    page_results.append(current)
                current = PageResult(page_number=0, year=line.split(":", 1)[1].strip())
            elif line.startswith("SUBJECT:") and not line.startswith("SUBJECT_ISSUES"):
                current.subject_obs = line.split(":", 1)[1].strip()
            elif line.startswith("SUBJECT_ISSUES:"):
                current.issues_noted["subject"] = line.split(":", 1)[1].strip()
            elif line.startswith("ADJOINING:") and not line.startswith("ADJOINING_ISSUES"):
                current.adjoining_obs = line.split(":", 1)[1].strip()
            elif line.startswith("ADJOINING_ISSUES:"):
                current.issues_noted["adjoining"] = line.split(":", 1)[1].strip()
            elif line.startswith("SURROUNDING:") and not line.startswith("SURROUNDING_ISSUES"):
                current.surrounding_obs = line.split(":", 1)[1].strip()
            elif line.startswith("SURROUNDING_ISSUES:"):
                current.issues_noted["surrounding"] = line.split(":", 1)[1].strip()
        
        if current.year:
            page_results.append(current)
        
        result.pages = page_results
        result.years_reviewed = sorted(set(p.year for p in page_results if p.year))
        result.tables = _build_zone_tables(page_results)
    
    if progress_callback:
        await progress_callback(f"Completed {filename}", total, total)
    
    return result


# --- Cross-referencing summary ---

SUMMARY_PROMPT = """You are writing the summary paragraph for the Historical Documentation section of a Phase I Environmental Site Assessment.

You have analyzed the following historical sources for this property:

{findings}

Write a professional ESA summary paragraph that:

1. States the date range of historical sources reviewed (e.g., "Historical sources reviewed for the subject property included aerial photographs from 1946 through 2024, topographic maps from 1949 through 2023, and city directory listings from 1995 through 2020.")

2. Summarizes the general development history of the subject property and surrounding area.

3. CROSS-REFERENCES findings between document types:
   - If a city directory lists an environmentally concerning use (e.g., gasoline service station) AND the aerial photograph shows visual evidence (fuel canopy, tanks), note that BOTH sources corroborate the concern.
   - If a city directory lists a concerning use (e.g., machine shop, auto repair) but the aerial photograph shows a normal-looking commercial building with no visible environmental indicators, note that the concern was identified in the city directory only.
   - If an aerial photograph shows visual environmental evidence (tanks, staining) but no city directory listing explains it, note that the concern was identified in aerial photographs only.

4. Clearly state whether any environmental concerns were identified, and if so, specify:
   - WHAT the concern is
   - WHERE it was identified (subject property, adjoining, or surrounding)
   - WHICH source(s) identified it (aerial photographs, topographic maps, city directories, or multiple)
   - WHAT time period it was observed

5. If no environmental concerns were identified across any source, state: "No environmental concerns were identified in the historical documentation reviewed."

Write in professional ESA tone. Be specific and factual. Do not speculate beyond what the data shows."""


async def generate_summary(documents: list, llm: LLMProvider) -> str:
    """Generate a cross-referencing summary across all document types."""
    findings_parts = []
    
    for doc in documents:
        doc_label = {
            "aerial": "Aerial Photographs",
            "topo": "Topographic Maps",
            "city_directory": "City Directories",
        }.get(doc.doc_type.value, doc.doc_type.value)
        
        years = ", ".join(doc.years_reviewed) if doc.years_reviewed else "unknown years"
        
        # Collect issues
        issues = []
        for table in doc.tables:
            zone_label = table.zone.value
            for row in table.rows:
                if row.issues_noted.lower().startswith("yes"):
                    issues.append(f"  - {zone_label.upper()} ({row.year_range}): {row.issues_noted}")
        
        part = f"**{doc_label}** (years: {years}):\n"
        if issues:
            part += "  Environmental concerns identified:\n" + "\n".join(issues)
        else:
            part += "  No environmental concerns identified."
        
        findings_parts.append(part)
    
    findings_text = "\n\n".join(findings_parts)
    prompt = SUMMARY_PROMPT.format(findings=findings_text)
    
    loop = asyncio.get_event_loop()
    summary = await loop.run_in_executor(_executor, llm.generate_text, prompt)
    return summary
