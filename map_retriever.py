"""
Map Retriever Module — Phase I ESA Historical Analyzer

Fetches environmental map data (soil, wetlands, flood zones) for a given
lat/lon coordinate using public government APIs. No API keys required.

APIs used:
  1. USDA Soil Data Access (WSS) — soil type, drainage, hydric rating
  2. USFWS National Wetlands Inventory (NWI) — wetland features
  3. FEMA National Flood Hazard Layer (NFHL) — flood zone designation

Usage:
    from map_retriever import get_map_data
    result = get_map_data(33.7490, -84.3880)
    print(result.soil.map_unit_name)
    print(result.flood.zone)
"""

import asyncio
import hashlib
import json
import logging
import math
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SoilResult:
    mukey: str
    map_unit_name: str           # e.g., "Hanford sandy loam, 0-2% slopes"
    component_name: str          # soil series
    component_pct: float         # dominant component percentage
    drainage_class: str          # "Well drained", "Poorly drained", etc.
    hydrologic_group: str        # "A", "B", "C", or "D"
    hydric_rating: str           # "Yes", "No", "Unranked"
    tax_class: str               # taxonomic classification
    corrosion_concrete: Optional[str] = None
    corrosion_steel: Optional[str] = None
    map_image_path: Optional[str] = None


@dataclass
class WetlandFeature:
    attribute_code: str          # NWI code e.g. "PEM1C", "R2UBH"
    wetland_type: str            # "Palustrine", "Riverine", etc.
    acres: Optional[float]
    system_name: str             # "Palustrine", "Riverine", "Lacustrine"
    class_name: str              # "Emergent", "Forested", "Scrub-Shrub"
    water_regime: str            # "Permanently Flooded", "Seasonally Flooded"


@dataclass
class WetlandResult:
    present: bool
    features: list               # list of WetlandFeature
    wetland_types: list          # unique types present
    attribute_codes: list
    map_image_path: Optional[str] = None


@dataclass
class FloodResult:
    zone: str                    # "AE", "X", "A", "VE", "D", "Unknown"
    zone_subtype: Optional[str]
    is_sfha: bool                # Special Flood Hazard Area (high risk)
    base_flood_elevation_ft: Optional[float]
    v_datum: Optional[str]       # vertical datum
    depth_ft: Optional[float]
    study_type: Optional[str]
    dfirm_panel: Optional[str]
    flood_risk_summary: str      # human-readable risk text
    map_image_path: Optional[str] = None


@dataclass
class MapResult:
    lat: float
    lon: float
    bbox: tuple                  # (xmin, ymin, xmax, ymax)
    radius_ft: float
    soil: Optional[SoilResult] = None
    wetland: Optional[WetlandResult] = None
    flood: Optional[FloodResult] = None
    errors: dict = field(default_factory=dict)   # {api_name: error_msg}
    cached: bool = False
    retrieved_at: str = ""       # ISO timestamp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def lat_lon_to_bbox(lat: float, lon: float, radius_ft: float = 1000) -> tuple:
    """Convert point + radius to WGS84 bounding box (xmin, ymin, xmax, ymax)."""
    radius_deg_lat = (radius_ft / 5280) / 69.0
    radius_deg_lon = radius_deg_lat / math.cos(math.radians(lat))
    return (
        round(lon - radius_deg_lon, 6),
        round(lat - radius_deg_lat, 6),
        round(lon + radius_deg_lon, 6),
        round(lat + radius_deg_lat, 6),
    )


def _flood_risk_summary(zone: str, is_sfha: bool, subtype: str = None) -> str:
    """Return a human-readable flood risk description."""
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


def _cache_key(lat: float, lon: float, radius_ft: float) -> str:
    """MD5 hash of rounded coordinates for cache lookup."""
    raw = f"{lat:.4f},{lon:.4f},{radius_ft:.0f}"
    return hashlib.md5(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Serialization helpers (dataclass <-> JSON-safe dict)
# ---------------------------------------------------------------------------

def _serialize_result(result: MapResult) -> str:
    """Serialize a MapResult to JSON string for caching."""
    d = asdict(result)
    # Convert tuple to list for JSON
    d["bbox"] = list(d["bbox"])
    return json.dumps(d)


def _deserialize_result(json_str: str) -> MapResult:
    """Deserialize a JSON string back into a MapResult."""
    d = json.loads(json_str)
    d["bbox"] = tuple(d["bbox"])

    # Reconstruct SoilResult
    if d.get("soil"):
        d["soil"] = SoilResult(**d["soil"])

    # Reconstruct WetlandResult with nested WetlandFeature objects
    if d.get("wetland"):
        wd = d["wetland"]
        wd["features"] = [WetlandFeature(**f) for f in wd.get("features", [])]
        d["wetland"] = WetlandResult(**wd)

    # Reconstruct FloodResult
    if d.get("flood"):
        d["flood"] = FloodResult(**d["flood"])

    return MapResult(**d)


# ---------------------------------------------------------------------------
# SQLite Cache
# ---------------------------------------------------------------------------

class _MapCache:
    """Simple SQLite cache with 30-day TTL."""

    TTL_DAYS = 30

    def __init__(self):
        db_path = Path.home() / ".esa_map_cache.db"
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS map_cache ("
            "  cache_key TEXT PRIMARY KEY,"
            "  data_json TEXT NOT NULL,"
            "  cached_at TEXT NOT NULL"
            ")"
        )
        self._conn.commit()

    def get(self, key: str) -> Optional[MapResult]:
        row = self._conn.execute(
            "SELECT data_json, cached_at FROM map_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None

        cached_at = datetime.fromisoformat(row[1])
        if datetime.now(timezone.utc) - cached_at > timedelta(days=self.TTL_DAYS):
            # Expired — delete and return miss
            self._conn.execute("DELETE FROM map_cache WHERE cache_key = ?", (key,))
            self._conn.commit()
            return None

        result = _deserialize_result(row[0])
        result.cached = True
        return result

    def put(self, key: str, result: MapResult) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO map_cache (cache_key, data_json, cached_at) "
            "VALUES (?, ?, ?)",
            (key, _serialize_result(result), now),
        )
        self._conn.commit()

    def close(self):
        self._conn.close()


# ---------------------------------------------------------------------------
# MapRetriever
# ---------------------------------------------------------------------------

class MapRetriever:
    """Async retriever for soil, wetland, and flood data."""

    API_TIMEOUT = aiohttp.ClientTimeout(total=30)

    def __init__(self, output_dir: str = "./map_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._cache = _MapCache()

    async def retrieve(
        self, lat: float, lon: float, radius_ft: float = 1000
    ) -> MapResult:
        """
        Retrieve soil, wetland, and flood data for a coordinate.

        Args:
            lat: Latitude (decimal degrees, WGS84)
            lon: Longitude (decimal degrees, WGS84)
            radius_ft: Search radius in feet (default 1000)

        Returns:
            MapResult with soil, wetland, and flood data (any may be None on error)
        """
        key = _cache_key(lat, lon, radius_ft)

        # Check cache
        cached = self._cache.get(key)
        if cached is not None:
            logger.info("Cache hit for %s", key)
            return cached

        bbox = lat_lon_to_bbox(lat, lon, radius_ft)
        result = MapResult(
            lat=lat,
            lon=lon,
            bbox=bbox,
            radius_ft=radius_ft,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
        )

        # Fire all 3 API groups in parallel
        async with aiohttp.ClientSession(timeout=self.API_TIMEOUT) as session:
            soil_task = self._fetch_soil(session, lat, lon, bbox)
            wetland_task = self._fetch_wetlands(session, bbox)
            flood_task = self._fetch_flood(session, lat, lon)

            outcomes = await asyncio.gather(
                soil_task, wetland_task, flood_task, return_exceptions=True
            )

        # Process soil result
        if isinstance(outcomes[0], Exception):
            err = str(outcomes[0])
            logger.error("Soil API error: %s", err)
            result.errors["soil"] = err
        else:
            result.soil = outcomes[0]

        # Process wetland result
        if isinstance(outcomes[1], Exception):
            err = str(outcomes[1])
            logger.error("Wetland API error: %s", err)
            result.errors["wetland"] = err
        else:
            result.wetland = outcomes[1]

        # Process flood result
        if isinstance(outcomes[2], Exception):
            err = str(outcomes[2])
            logger.error("Flood API error: %s", err)
            result.errors["flood"] = err
        else:
            result.flood = outcomes[2]

        # Cache the result
        self._cache.put(key, result)
        return result

    # ------------------------------------------------------------------
    # 1. USDA Soil Data Access
    # ------------------------------------------------------------------

    async def _fetch_soil(
        self, session: aiohttp.ClientSession, lat: float, lon: float, bbox: tuple
    ) -> Optional[SoilResult]:
        """Fetch soil data from USDA SDA (tabular + WMS map)."""

        # --- Tabular query ---
        sql = (
            "SELECT mu.mukey, mu.muname, c.compname, c.comppct_r, "
            "c.taxclname, c.drainagecl, c.hydgrp, c.hydricrating, "
            "c.corcon, c.corsteel "
            "FROM mapunit mu "
            "INNER JOIN component c ON mu.mukey = c.mukey "
            "WHERE mu.mukey IN ("
            "  SELECT mukey FROM SDA_Get_Mukey_from_intersection_with_WktWgs84("
            f"    'POINT({lon} {lat})'"
            "  )"
            ") AND c.majcompflag = 'Yes' "
            "ORDER BY c.comppct_r DESC"
        )

        tabular_url = (
            "https://SDMDataAccess.sc.egov.usda.gov"
            "/Tabular/SDMTabularService/post.rest"
        )
        payload = {"REQUEST": "query", "QUERY": sql, "FORMAT": "JSON"}

        async with session.post(tabular_url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json(content_type=None)

        table = data.get("Table")
        if not table or len(table) == 0:
            logger.warning("Soil query returned no data for (%s, %s)", lat, lon)
            return None

        # Take the first (dominant) row
        row = table[0]
        # Column order: mukey, muname, compname, comppct_r, taxclname,
        #               drainagecl, hydgrp, hydricrating, corcon, corsteel
        soil = SoilResult(
            mukey=str(row[0] or ""),
            map_unit_name=str(row[1] or ""),
            component_name=str(row[2] or ""),
            component_pct=float(row[3]) if row[3] is not None else 0.0,
            tax_class=str(row[4] or ""),
            drainage_class=str(row[5] or ""),
            hydrologic_group=str(row[6] or ""),
            hydric_rating=str(row[7] or ""),
            corrosion_concrete=str(row[8]) if row[8] is not None else None,
            corrosion_steel=str(row[9]) if row[9] is not None else None,
        )

        # --- WMS map image ---
        try:
            soil.map_image_path = await self._fetch_soil_map(session, bbox)
        except Exception as e:
            logger.warning("Soil map image fetch failed: %s", e)

        return soil

    async def _fetch_soil_map(
        self, session: aiohttp.ClientSession, bbox: tuple
    ) -> Optional[str]:
        """Download the USDA WMS soil map image."""
        xmin, ymin, xmax, ymax = bbox
        url = "https://SDMDataAccess.sc.egov.usda.gov/Spatial/SDM.wms"
        params = {
            "SERVICE": "WMS",
            "VERSION": "1.1.1",
            "REQUEST": "GetMap",
            "LAYERS": "SurveyAreaPoly,MapunitPoly",
            "STYLES": "",
            "SRS": "EPSG:4326",
            "BBOX": f"{xmin},{ymin},{xmax},{ymax}",
            "WIDTH": "800",
            "HEIGHT": "600",
            "FORMAT": "image/png",
            "TRANSPARENT": "TRUE",
        }

        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            img_data = await resp.read()

        out_path = self.output_dir / "soil_map.png"
        out_path.write_bytes(img_data)
        logger.info("Soil map saved to %s", out_path)
        return str(out_path)

    # ------------------------------------------------------------------
    # 2. USFWS National Wetlands Inventory
    # ------------------------------------------------------------------

    async def _fetch_wetlands(
        self, session: aiohttp.ClientSession, bbox: tuple
    ) -> WetlandResult:
        """Fetch wetland features and map from NWI."""

        xmin, ymin, xmax, ymax = bbox

        # --- Feature query ---
        query_url = (
            "https://fwspublicservices.wim.usgs.gov"
            "/wetlandsmapservice/rest/services/Wetlands/MapServer/0/query"
        )
        params = {
            "where": "1=1",
            "geometry": f"{xmin},{ymin},{xmax},{ymax}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "returnGeometry": "false",
            "f": "json",
        }

        async with session.get(query_url, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json(content_type=None)

        raw_features = data.get("features", [])
        features = []
        wetland_types = set()
        attribute_codes = []

        for feat in raw_features:
            attrs = feat.get("attributes", {})
            code = (
                attrs.get("Wetlands.ATTRIBUTE")
                or attrs.get("ATTRIBUTE")
                or ""
            )
            wtype = (
                attrs.get("Wetlands.WETLAND_TYPE")
                or attrs.get("WETLAND_TYPE")
                or ""
            )
            acres = (
                attrs.get("Wetlands.ACRES")
                or attrs.get("ACRES")
            )
            system_name = (
                attrs.get("NWI_Wetland_Codes.SYSTEM_NAME")
                or attrs.get("SYSTEM_NAME")
                or ""
            )
            class_name = (
                attrs.get("NWI_Wetland_Codes.CLASS_NAME")
                or attrs.get("CLASS_NAME")
                or ""
            )
            water_regime = (
                attrs.get("NWI_Wetland_Codes.WATER_REGIME_NAME")
                or attrs.get("WATER_REGIME_NAME")
                or ""
            )

            wf = WetlandFeature(
                attribute_code=code,
                wetland_type=wtype,
                acres=float(acres) if acres is not None else None,
                system_name=system_name,
                class_name=class_name,
                water_regime=water_regime,
            )
            features.append(wf)
            if wtype:
                wetland_types.add(wtype)
            if code:
                attribute_codes.append(code)

        result = WetlandResult(
            present=len(features) > 0,
            features=features,
            wetland_types=sorted(wetland_types),
            attribute_codes=attribute_codes,
        )

        # --- WMS map image ---
        try:
            result.map_image_path = await self._fetch_wetland_map(session, bbox)
        except Exception as e:
            logger.warning("Wetland map image fetch failed: %s", e)

        return result

    async def _fetch_wetland_map(
        self, session: aiohttp.ClientSession, bbox: tuple
    ) -> Optional[str]:
        """Download the NWI WMS wetland map image."""
        xmin, ymin, xmax, ymax = bbox
        url = (
            "https://fwspublicservices.wim.usgs.gov"
            "/wetlandsmapservice/services/Wetlands/MapServer/WMSServer"
        )
        params = {
            "REQUEST": "GetMap",
            "SERVICE": "WMS",
            "VERSION": "1.1.1",
            "LAYERS": "0",
            "STYLES": "",
            "SRS": "EPSG:4326",
            "BBOX": f"{xmin},{ymin},{xmax},{ymax}",
            "WIDTH": "800",
            "HEIGHT": "600",
            "FORMAT": "image/png",
            "TRANSPARENT": "TRUE",
        }

        async with session.get(url, params=params) as resp:
            resp.raise_for_status()
            img_data = await resp.read()

        out_path = self.output_dir / "wetlands_map.png"
        out_path.write_bytes(img_data)
        logger.info("Wetland map saved to %s", out_path)
        return str(out_path)

    # ------------------------------------------------------------------
    # 3. FEMA National Flood Hazard Layer
    # ------------------------------------------------------------------

    async def _fetch_flood(
        self, session: aiohttp.ClientSession, lat: float, lon: float
    ) -> FloodResult:
        """Fetch flood zone data from FEMA NFHL via ArcGIS."""

        query_url = (
            "https://services.arcgis.com/P3ePLMYs2RVChkJx/ArcGIS/rest/services"
            "/USA_Flood_Hazard_Reduced_Set_gdb/FeatureServer/0/query"
        )
        params = {
            "where": "1=1",
            "geometry": f"{lon},{lat}",
            "geometryType": "esriGeometryPoint",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": (
                "FLD_ZONE,ZONE_SUBTY,SFHA_TF,STATIC_BFE,"
                "V_DATUM,DEPTH,STUDY_TYP,DFIRM_ID"
            ),
            "returnGeometry": "false",
            "f": "json",
        }

        async with session.get(query_url, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json(content_type=None)

        raw_features = data.get("features", [])

        if not raw_features:
            return FloodResult(
                zone="Unknown",
                zone_subtype=None,
                is_sfha=False,
                base_flood_elevation_ft=None,
                v_datum=None,
                depth_ft=None,
                study_type=None,
                dfirm_panel=None,
                flood_risk_summary=_flood_risk_summary("Unknown", False),
            )

        attrs = raw_features[0].get("attributes", {})

        zone = attrs.get("FLD_ZONE") or "Unknown"
        subtype = attrs.get("ZONE_SUBTY")
        sfha_raw = attrs.get("SFHA_TF", "F")
        is_sfha = sfha_raw == "T"
        bfe = attrs.get("STATIC_BFE")
        v_datum = attrs.get("V_DATUM")
        depth = attrs.get("DEPTH")
        study_type = attrs.get("STUDY_TYP")
        dfirm = attrs.get("DFIRM_ID")

        return FloodResult(
            zone=zone,
            zone_subtype=subtype if subtype else None,
            is_sfha=is_sfha,
            base_flood_elevation_ft=float(bfe) if bfe is not None else None,
            v_datum=str(v_datum) if v_datum is not None else None,
            depth_ft=float(depth) if depth is not None else None,
            study_type=str(study_type) if study_type is not None else None,
            dfirm_panel=str(dfirm) if dfirm is not None else None,
            flood_risk_summary=_flood_risk_summary(zone, is_sfha, subtype),
        )

    def close(self):
        """Close the cache connection."""
        self._cache.close()


# ---------------------------------------------------------------------------
# Sync convenience wrapper
# ---------------------------------------------------------------------------

def get_map_data(
    lat: float,
    lon: float,
    radius_ft: float = 1000,
    output_dir: str = "./map_output",
) -> MapResult:
    """
    Synchronous convenience wrapper around MapRetriever.retrieve().

    Args:
        lat: Latitude (decimal degrees, WGS84)
        lon: Longitude (decimal degrees, WGS84)
        radius_ft: Search radius in feet (default 1000)
        output_dir: Directory for map image output (default ./map_output/)

    Returns:
        MapResult with soil, wetland, and flood data
    """
    retriever = MapRetriever(output_dir=output_dir)
    try:
        # Handle the case where an event loop is already running
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an existing event loop (e.g., Jupyter, async server)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(
                    asyncio.run, retriever.retrieve(lat, lon, radius_ft)
                ).result()
        else:
            result = asyncio.run(retriever.retrieve(lat, lon, radius_ft))

        return result
    finally:
        retriever.close()


# ---------------------------------------------------------------------------
# CLI / standalone usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if len(sys.argv) >= 3:
        lat_arg = float(sys.argv[1])
        lon_arg = float(sys.argv[2])
        radius_arg = float(sys.argv[3]) if len(sys.argv) > 3 else 1000
    else:
        # Default: downtown Atlanta
        lat_arg = 33.7490
        lon_arg = -84.3880
        radius_arg = 1000

    print(f"\nFetching map data for ({lat_arg}, {lon_arg}), radius={radius_arg} ft...\n")
    map_result = get_map_data(lat_arg, lon_arg, radius_arg)

    if map_result.soil:
        s = map_result.soil
        print("=== SOIL ===")
        print(f"  Map Unit:        {s.map_unit_name}")
        print(f"  Component:       {s.component_name} ({s.component_pct}%)")
        print(f"  Drainage:        {s.drainage_class}")
        print(f"  Hydrologic Grp:  {s.hydrologic_group}")
        print(f"  Hydric:          {s.hydric_rating}")
        print(f"  Taxonomy:        {s.tax_class}")
        if s.corrosion_concrete:
            print(f"  Corrosion (conc): {s.corrosion_concrete}")
        if s.corrosion_steel:
            print(f"  Corrosion (steel):{s.corrosion_steel}")
        if s.map_image_path:
            print(f"  Map image:       {s.map_image_path}")
        print()

    if map_result.wetland:
        w = map_result.wetland
        print("=== WETLANDS ===")
        print(f"  Present:         {w.present}")
        print(f"  Types:           {', '.join(w.wetland_types) if w.wetland_types else 'None'}")
        print(f"  Feature count:   {len(w.features)}")
        for wf in w.features[:5]:
            print(f"    - {wf.attribute_code}: {wf.wetland_type} "
                  f"({wf.class_name}, {wf.water_regime})"
                  f"{f', {wf.acres:.1f} ac' if wf.acres else ''}")
        if len(w.features) > 5:
            print(f"    ... and {len(w.features) - 5} more")
        if w.map_image_path:
            print(f"  Map image:       {w.map_image_path}")
        print()

    if map_result.flood:
        f = map_result.flood
        print("=== FLOOD ===")
        print(f"  Zone:            {f.zone}")
        print(f"  SFHA:            {f.is_sfha}")
        print(f"  Risk:            {f.flood_risk_summary}")
        if f.zone_subtype:
            print(f"  Subtype:         {f.zone_subtype}")
        if f.base_flood_elevation_ft is not None:
            print(f"  Base elev (ft):  {f.base_flood_elevation_ft}")
        if f.dfirm_panel:
            print(f"  DFIRM panel:     {f.dfirm_panel}")
        print()

    if map_result.errors:
        print("=== ERRORS ===")
        for api, err in map_result.errors.items():
            print(f"  {api}: {err}")
        print()

    print(f"Retrieved at: {map_result.retrieved_at}")
    print(f"Cached: {map_result.cached}")
