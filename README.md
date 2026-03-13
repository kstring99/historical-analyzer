# Historical Documentation Analyzer

Phase I ESA tool that analyzes ERIS historical documentation packages (aerial photographs, topographic maps, city directories) and produces formatted tables with cross-referencing summary.

## Features

- **Aerial Photograph Analysis** — AI vision identifies land use, structures, and environmental concerns within the property boundary
- **Topographic Map Analysis** — Extrapolates property location, describes terrain and features
- **City Directory Parsing** — Extracts occupant listings, flags environmentally concerning uses
- **Cross-Reference Summary** — Synthesizes findings across all document types, identifies RECs
- **Three-zone analysis** — Subject Property, Adjoining Properties, Surrounding Properties
- **Year range grouping** — Consecutive similar years automatically grouped
- **Model agnostic** — Works with OpenAI (GPT-4o) or Anthropic (Claude Sonnet)

## Prerequisites

- Python 3.10+
- **poppler** (required for PDF processing):
  - macOS: `brew install poppler`
  - Ubuntu/Debian: `apt-get install poppler-utils`
  - Windows: [Download poppler](https://github.com/oschwartz10612/poppler-windows/releases)
  - Or just use Docker (see below) — poppler is included.

## Quick Start (Local)

```bash
# Clone the repo
git clone https://github.com/kstring99/historical-analyzer.git
cd historical-analyzer

# Install dependencies
pip install -r requirements.txt

# Set your API key (OpenAI or Anthropic — pick one)
export OPENAI_API_KEY=sk-...
# or: export ANTHROPIC_API_KEY=sk-ant-...

# Run
uvicorn app.main:app --port 8000
```

Open http://localhost:8000

## Docker Deployment

```bash
# Build
docker build -t esa-historical-analyzer .

# Run with OpenAI enterprise key
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... esa-historical-analyzer
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key (enterprise) | Yes (for OpenAI) |
| `ANTHROPIC_API_KEY` | Anthropic API key | Yes (for Anthropic) |

## Output Format

Produces three tables per document type matching the Quire ESA template:
- **Year** — Individual year or grouped range (e.g., "1946 - 1978")
- **Issues Noted** — Yes/No with specific evidence cited
- **Observations/Occupants** — Professional ESA-style descriptions

Plus a cross-referencing summary paragraph that synthesizes findings across all sources.
