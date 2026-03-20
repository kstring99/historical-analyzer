# Claude Code Prompt — Map Retriever Module

**File:** `~/EBI/historical-analyzer/CC_MAP_RETRIEVER_PROMPT.md`  
**Use:** Paste this entire prompt to Claude Code to build the map_retriever module

---

## Prompt

Build a `map_retriever.py` module for the Phase I ESA Historical Analyzer. This tool fetches environmental map data deterministically (no LLM) for a given lat/lon coordinate. Read the API spec at `~/clawd/research/map-api-spec.md` first.

### What to Build

Create `~/EBI/historical-analyzer/map_retriever.py` with:

1. **`MapRetriever` class** with async `retrieve(lat, lon, radius_ft=1000)` method
2. **`get_map_data(lat, lon, radius_ft=1000)`** sync convenience wrapper
3. **Structured output** as dataclasses: `MapResult`, `SoilResult`, `WetlandResult`, `FloodResult`
4. **SQLite cache** at `~/.esa_map_cache.db` with 30-day TTL
5. **Map images** saved to `./map_output/` (soil_map.png, wetlands_map.png)
6. **Parallel execution** via `asyncio.gather` — all 3 APIs fire simultaneously

### The Three APIs (all confirmed working, no auth required)

#### 1. USDA Soil Data Access (WSS)
- **Tabular:** `POST https://SDMDataAccess.sc.egov.usda.gov/Tabular/SDMTabularService/post.rest`
  - Body: `{"REQUEST": "query", "QUERY": "<SQL>", "FORMAT": "JSON"}`
  - SQL: `SELECT mu.mukey, mu.muname, c.compname, c.comppct_r, c.taxclname, c.drainagecl, c.hydgrp, c.hydricrating, c.corcon, c.corsteel FROM mapunit mu INNER JOIN component c ON mu.mukey = c.mukey WHERE mu.mukey IN (SELECT mukey FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('POINT({lon} {lat})')) AND c.majcompflag = 'Yes' ORDER BY c.comppct_r DESC`
  - ⚠️ WKT is `POINT(lon lat)` — longitude FIRST
  - Response: `{"Table": [[row1_values], [row2_values], ...]}`
  - Column order: mukey, muname, compname, comppct_r, taxclname, drainagecl, hydgrp, hydricrating, corcon, corsteel

- **WMS Map:** `GET https://SDMDataAccess.sc.egov.usda.gov/Spatial/SDM.wms`
  - Params: `SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS=SurveyAreaPoly,MapunitPoly&STYLES=&SRS=EPSG:4326&BBOX={xmin},{ymin},{xmax},{ymax}&WIDTH=800&HEIGHT=600&FORMAT=image/png&TRANSPARENT=TRUE`

#### 2. USFWS National Wetlands Inventory (NWI)
- **Feature query:** `GET https://fwspublicservices.wim.usgs.gov/wetlandsmapservice/rest/services/Wetlands/MapServer/0/query`
  - Params: `where=1=1&geometry={xmin},{ymin},{xmax},{ymax}&geometryType=esriGeometryEnvelope&inSR=4326&spatialRel=esriSpatialRelIntersects&outFields=*&returnGeometry=false&f=json`
  - ⚠️ `where=1=1` is REQUIRED — API returns 400 without it
  - Response field names have table prefixes: `Wetlands.ATTRIBUTE`, `Wetlands.WETLAND_TYPE`, `NWI_Wetland_Codes.SYSTEM_NAME`, `NWI_Wetland_Codes.CLASS_NAME`, `NWI_Wetland_Codes.WATER_REGIME_NAME`

- **WMS Map:** `GET https://fwspublicservices.wim.usgs.gov/wetlandsmapservice/services/Wetlands/MapServer/WMSServer`
  - Params: `REQUEST=GetMap&SERVICE=WMS&VERSION=1.1.1&LAYERS=0&STYLES=&SRS=EPSG:4326&BBOX={xmin},{ymin},{xmax},{ymax}&WIDTH=800&HEIGHT=600&FORMAT=image/png&TRANSPARENT=TRUE`
  - ⚠️ `STYLES=` (empty string) is REQUIRED — omitting returns 400

#### 3. FEMA National Flood Hazard Layer (NFHL)
- **Feature query:** `GET https://services.arcgis.com/P3ePLMYs2RVChkJx/ArcGIS/rest/services/USA_Flood_Hazard_Reduced_Set_gdb/FeatureServer/0/query`
  - ⚠️ Do NOT use `hazards.fema.gov/gis/nfhl` — it returns 404 (CDN-blocked)
  - Params: `where=1=1&geometry={lon},{lat}&geometryType=esriGeometryPoint&inSR=4326&spatialRel=esriSpatialRelIntersects&outFields=FLD_ZONE,ZONE_SUBTY,SFHA_TF,STATIC_BFE,V_DATUM,DEPTH,STUDY_TYP,DFIRM_ID&returnGeometry=false&f=json`
  - ⚠️ ArcGIS geometry is `lon,lat` (longitude first)
  - Response: single feature with `FLD_ZONE`, `ZONE_SUBTY`, `SFHA_TF` ("T"=high risk, "F"=low risk), `STATIC_BFE`
  - Empty features = no FIRM map data, return zone="Unknown"

### Dataclass Definitions

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SoilResult:
    mukey: str
    map_unit_name: str       # e.g., "Hanford sandy loam, 0-2% slopes"
    component_name: str      # soil series
    component_pct: float     # dominant component percentage
    drainage_class: str      # "Well drained", "Poorly drained", etc.
    hydrologic_group: str    # "A", "B", "C", or "D"
    hydric_rating: str       # "Yes", "No", "Unranked"
    tax_class: str           # taxonomic classification
    corrosion_concrete: Optional[str] = None
    corrosion_steel: Optional[str] = None
    map_image_path: Optional[str] = None

@dataclass
class WetlandFeature:
    attribute_code: str      # NWI code e.g. "PEM1C", "R2UBH"
    wetland_type: str        # "Palustrine", "Riverine", etc.
    acres: Optional[float]
    system_name: str         # "Palustrine", "Riverine", "Lacustrine"
    class_name: str          # "Emergent", "Forested", "Scrub-Shrub"
    water_regime: str        # "Permanently Flooded", "Seasonally Flooded"

@dataclass
class WetlandResult:
    present: bool
    features: list            # list of WetlandFeature
    wetland_types: list[str]  # unique types present
    attribute_codes: list[str]
    map_image_path: Optional[str] = None

@dataclass
class FloodResult:
    zone: str                 # "AE", "X", "A", "VE", "D", "Unknown"
    zone_subtype: Optional[str]
    is_sfha: bool             # Special Flood Hazard Area (high risk)
    base_flood_elevation_ft: Optional[float]
    v_datum: Optional[str]    # vertical datum
    depth_ft: Optional[float]
    study_type: Optional[str]
    dfirm_panel: Optional[str]
    flood_risk_summary: str   # human-readable risk text
    map_image_path: Optional[str] = None

@dataclass
class MapResult:
    lat: float
    lon: float
    bbox: tuple               # (xmin, ymin, xmax, ymax)
    radius_ft: float
    soil: Optional[SoilResult] = None
    wetland: Optional[WetlandResult] = None
    flood: Optional[FloodResult] = None
    errors: dict = field(default_factory=dict)  # {api_name: error_msg}
    cached: bool = False
    retrieved_at: str = ""    # ISO timestamp
```

### Flood Risk Summary Helper
```python
def _flood_risk_summary(zone: str, is_sfha: bool, subtype: str = None) -> str:
    if zone in ("VE", "V"):
        return "Very High — Coastal High Hazard with wave action (Zone VE/V)"
    elif zone == "AO":
        return "High — Sheet flow flooding 1-3 ft depth (Zone AO)"
    elif zone == "AH":
        return "High — Shallow ponding flood hazard (Zone AH)"
    elif is_sfha:
        return f"High — Special Flood Hazard Area (Zone {zone})"
    elif subtype and "0.2" in subtype:
        return "Moderate — 500-year flood zone (0.2% annual chance)"
    elif zone == "X":
        return "Low — Minimal flood hazard (Zone X)"
    elif zone == "D":
        return "Undetermined — No FIRM study available"
    elif zone == "Unknown":
        return "Unknown — No FIRM panel data found"
    return f"See FIRM panel — Zone {zone}"
```

### Bounding Box Helper
```python
import math

def lat_lon_to_bbox(lat: float, lon: float, radius_ft: float = 1000) -> tuple:
    """Convert point + radius to WGS84 bounding box (xmin, ymin, xmax, ymax)."""
    radius_deg_lat = (radius_ft / 5280) / 69.0
    radius_deg_lon = radius_deg_lat / math.cos(math.radians(lat))
    return (
        round(lon - radius_deg_lon, 6),
        round(lat - radius_deg_lat, 6),
        round(lon + radius_deg_lon, 6),
        round(lat + radius_deg_lat, 6)
    )
```

### Caching
Use SQLite. Table: `map_cache(cache_key TEXT PRIMARY KEY, data_json TEXT, cached_at TEXT)`.  
Cache key: MD5 of `f"{lat:.4f},{lon:.4f},{radius_ft:.0f}"` (rounds to ~10m precision).  
TTL: 30 days. On miss, fetch live; on hit, deserialize MapResult from JSON.

### Error Handling
- Each API call wrapped in try/except
- On failure, log to `result.errors[api_name]` — don't crash the whole call
- `asyncio.gather(return_exceptions=True)` so one API failure doesn't block others
- Timeout: 30 seconds per API
- If FEMA returns empty features list → `FloodResult(zone="Unknown", is_sfha=False, ...)`

### Also Create: `test_map_retriever.py`
Test with these known coordinates:
```python
# San Diego (urban, Zone X, no wetlands) 
test1 = (32.7157, -117.1611)

# New Orleans (Zone X/AE, wetlands present)
test2 = (29.9511, -90.0715)

# Houston (urban, mixed zones)
test3 = (29.7604, -95.3698)
```

Print all results in a clean format. Check that:
- WSS returns soil data for all 3
- NWI returns wetland features for New Orleans
- FEMA returns a zone code for all 3
- Map images are saved as PNG files

### File Structure to Create
```
~/EBI/historical-analyzer/
├── map_retriever.py          ← main module
├── test_map_retriever.py     ← test script  
└── map_output/               ← auto-created, map PNGs go here
```

### Requirements
```
pip install aiohttp requests  
# Already in the project's requirements if they exist
```

### Notes
- No API keys or auth for any of these services
- All three APIs are free federal government data
- WSS and NWI confirmed working as of 2026-03-19
- FEMA ArcGIS Online layer confirmed working; `hazards.fema.gov` is blocked
- NWI service at old URL `www.fws.gov/wetlands/arcgis/` may be stale — use `fwspublicservices.wim.usgs.gov`
