# Historical Documentation Analyzer — ESA Tool

## Purpose
Web app that ingests ERIS historical documentation PDFs (aerial photographs, topographic maps, city directories) and produces formatted tables for Phase I Environmental Site Assessments (ESAs).

## Input
Three types of ERIS PDF packages (see samples in `~/EBI/Historicals/`):

### 1. Aerial Photographs (`US_Aerial.pdf` — 21 pages)
- Page 1: ERIS cover page with metadata (Project Property, Coordinates, Location, Project No, Order No)
- Pages 2+: One aerial photo per page, each showing a different year
- **Green rectangle** marks the subject property boundary on each aerial
- Footer bar has: Year, Source, Scale, Address, Order No
- Vision model should describe what's WITHIN the green boundary (land use, structures, development)

### 2. Topographic Maps (`US_Topo.pdf` — 11 pages)
- Page 1: ERIS cover page (same format as aerials)
- Pages 2+: USGS 7.5-minute topo map excerpts, one per year
- **NO property boundary marked** — must extrapolate from aerial data
- Use coordinates from cover page + property dimensions from aerial green box
- Footer has: Year, quad name, scale
- Vision model should describe terrain, structures, water features, land use at the property location

### 3. City Directories (`US_CD.pdf` — 24 pages)
- Page 1: ERIS cover page
- Pages 2+: Directory listings organized by Year + Street Name headers
- Each entry: Street Number | Occupant Name | Use Classification (e.g., RESIDENTIAL, AUTO REPAIR, etc.)
- Source line: "SOURCE: DIGITAL BUSINESS DIRECTORY"
- This is primarily OCR/text extraction, not vision

## Output
For each document type, produce a **formatted table**:

| Year/Range | Description |
|---|---|
| 1953 | The subject property appears to be undeveloped land with native vegetation. Surrounding properties include... |
| 1966 | The subject property appears to contain a single-story commercial structure with a paved parking area... |

- Aerial photographs → one table
- Topographic maps → one table  
- City directories → one table (may include address-level detail)

### Description Language Style
- Always start with "The subject property appears to..."
- Professional ESA tone — factual, observational
- Note changes from previous year
- Flag anything environmentally relevant (tanks, industrial use, staining, cleared land, water features)
- For city directories: note business types, especially environmentally concerning ones (gas stations, dry cleaners, auto repair, chemical storage, etc.)

## Architecture

### Stack
- **Backend:** Python FastAPI
- **Frontend:** Clean HTML/CSS/JS (no framework — keep it simple and fast to ship)
- **PDF Processing:** pdf2image (poppler) for page extraction
- **LLM:** Model-agnostic abstraction layer

### LLM Abstraction
```python
class LLMProvider:
    """Abstract interface — swap models via config"""
    def analyze_image(self, image_bytes, prompt) -> str
    def generate_text(self, prompt) -> str

class AnthropicProvider(LLMProvider): ...
class OpenAIProvider(LLMProvider): ...
```

Default to Anthropic (Claude Sonnet) since we have API access. OpenAI support for when this gets deployed on the company's enterprise subscription.

### API Endpoints
```
POST /api/upload          — Upload PDF(s), returns job_id
GET  /api/status/{job_id} — Check processing status
GET  /api/results/{job_id} — Get formatted tables
POST /api/export/{job_id}  — Export as HTML/Word table
```

### Processing Pipeline
1. **Upload PDF** → extract pages as images (pdf2image, 200 DPI)
2. **Parse cover page** → extract metadata (coordinates, address, project name) via vision or OCR
3. **Classify document type** → aerial/topo/city directory (from filename or cover page content)
4. **For each page after cover:**
   - **Aerials:** Send image to vision model with prompt: "Describe what is WITHIN the green rectangle boundary on this aerial photograph. What land use, structures, development do you see? What year is shown in the footer?"
   - **Topos:** Send image to vision model with prompt: "The subject property is located at [coords], approximately [X x Y feet], near [road references from aerial]. Describe the terrain, structures, water features, and land use at that location on this topographic map. What year is shown?"
   - **City Dirs:** OCR the page, parse year/street/entries, flag non-residential uses
5. **Generate table** — LLM formats observations into ESA prose style
6. **Return results** — formatted HTML tables

### Frontend UI
- Clean, professional design (this will be shown to an ESA technical director)
- Drag-and-drop PDF upload area
- Progress indicator during processing
- Results displayed as formatted tables
- Export buttons (Copy to clipboard, Download as HTML)
- Settings panel for LLM provider selection

## Sample Data
Test PDFs are at `~/EBI/Historicals/`:
- `US_Aerial.pdf` (10.4 MB, 21 pages)
- `US_Topo.pdf` (4.5 MB, 11 pages)
- `US_CD.pdf` (517 KB, 24 pages)

## Key Technical Notes
- poppler is installed (`brew install poppler` done)
- Dependencies: pdf2image, Pillow, requests (see requirements.txt)
- For Anthropic: use Claude Sonnet for vision analysis (cost-effective, good at image description)
- Images should be sent as base64 to the vision API
- Process pages in parallel where possible (batch API calls)

## Ship Criteria
- [ ] Upload 3 PDFs → get 3 formatted tables
- [ ] Tables match ESA professional style
- [ ] Clean UI that looks good in a demo
- [ ] Model-agnostic (config switch between Anthropic/OpenAI)
- [ ] Works on localhost for demo today
