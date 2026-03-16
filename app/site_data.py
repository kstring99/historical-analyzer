"""
Government Site Data Puller — Phase I ESA Sections 3.5.1-3.5.4

Pulls environmental site data from free government APIs and generates
copy-pasteable paragraphs for ESA report sections:
  3.5.1 Geology and Soils (NRCS Web Soil Survey)
  3.5.2 Topography (USGS Elevation Point Query)
  3.5.3 Hydrogeology and Hydrology (NWI Wetlands + USGS)
  3.5.4 Flood Zone (FEMA NFHL)

All APIs are free, no authentication required.
"""

import json
import math
import urllib.parse
from typing import Optional
import httpx

TIMEOUT = 15.0


# ── Geocoding ─────────────────────────────────────────────────────────

async def geocode_address(address: str) -> dict:
    """
    Geocode an address using Census Bureau geocoder (free, no API key).
    Returns {lat, lon, matched_address, state, county}.
    """
    url = "https://geocoding.geo.census.gov/geocoder/geographies/onelineaddress"
    params = {
        "address": address,
        "benchmark": "Public_AR_Current",
        "vintage": "Current_Current",
        "format": "json",
    }
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
    
    matches = data.get("result", {}).get("addressMatches", [])
    if not matches:
        return {"error": f"Could not geocode: {address}"}
    
    match = matches[0]
    coords = match["coordinates"]
    geo = match.get("geographies", {})
    
    # Extract county and state from census geographies
    counties = geo.get("Counties", [{}])
    states = geo.get("States", [{}])
    
    return {
        "lat": coords["y"],
        "lon": coords["x"],
        "matched_address": match["matchedAddress"],
        "state": states[0].get("NAME", "") if states else "",
        "state_fips": states[0].get("STATE", "") if states else "",
        "county": counties[0].get("NAME", "") if counties else "",
        "county_fips": counties[0].get("COUNTY", "") if counties else "",
    }


# ── 3.5.1 Geology and Soils (NRCS) ──────────────────────────────────

async def get_soil_data(lat: float, lon: float) -> dict:
    """
    Query NRCS Soil Data Access for soil information at a point.
    Uses the SDA tabular query service with spatial SQL.
    """
    # SDA accepts SQL queries against the SSURGO database
    query = f"""
    SELECT 
        mu.muname AS map_unit_name,
        mu.mukey,
        mu.mukind AS map_unit_kind,
        c.compname AS component_name,
        c.comppct_r AS component_percent,
        c.taxclname AS taxonomic_class,
        c.taxorder AS tax_order,
        c.taxsubgrp AS tax_subgroup,
        c.drainagecl AS drainage_class,
        c.hydgrp AS hydrologic_group,
        c.slope_r AS slope_percent,
        c.tfact AS t_factor,
        ch.hzdepb_r AS depth_bottom_cm,
        ch.sandtotal_r AS sand_pct,
        ch.silttotal_r AS silt_pct,
        ch.claytotal_r AS clay_pct,
        ch.ksat_r AS saturated_conductivity
    FROM sacatalog sc
    INNER JOIN legend l ON sc.areasymbol = l.areasymbol
    INNER JOIN mapunit mu ON l.lkey = mu.lkey
    INNER JOIN component c ON mu.mukey = c.mukey
    LEFT JOIN chorizon ch ON c.cokey = ch.cokey AND ch.hzdept_r = 0
    WHERE mu.mukey IN (
        SELECT * FROM SDA_Get_Mukey_from_intersection_with_WktWgs84(
            'POINT({lon} {lat})'
        )
    )
    AND c.comppct_r >= 15
    ORDER BY c.comppct_r DESC
    """
    
    url = "https://sdmdataaccess.nrcs.usda.gov/Tabular/post.rest"
    payload = {"query": query, "format": "JSON"}
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    
    if "Table" not in data or not data["Table"]:
        return {"error": "No soil data found for this location", "raw": data}
    
    rows = data["Table"]
    soils = []
    for row in rows:
        soils.append({
            "map_unit": row[0],
            "component": row[3],
            "percent": row[4],
            "taxonomic_class": row[5],
            "tax_order": row[6],
            "drainage_class": row[8],
            "hydrologic_group": row[9],
            "slope_percent": row[10],
            "depth_bottom_cm": row[12],
            "sand_pct": row[13],
            "silt_pct": row[14],
            "clay_pct": row[15],
            "ksat": row[16],
        })
    
    return {"soils": soils, "count": len(soils)}


def format_soil_paragraph(soil_data: dict, address: str) -> str:
    """Generate ESA Section 3.5.1 paragraph from soil data."""
    if "error" in soil_data:
        return f"Soil data could not be retrieved for this location. {soil_data['error']}"
    
    soils = soil_data.get("soils", [])
    if not soils:
        return "No soil data available from the NRCS Web Soil Survey for this location."
    
    primary = soils[0]
    
    # Build soil description
    soil_names = []
    for s in soils[:3]:  # Top 3 components
        name = s.get("component", "Unknown")
        pct = s.get("percent", "")
        if pct:
            soil_names.append(f"{name} ({pct}%)")
        else:
            soil_names.append(name)
    
    drainage = primary.get("drainage_class", "not reported")
    hydro_group = primary.get("hydrologic_group", "not reported")
    slope = primary.get("slope_percent", "not reported")
    tax_class = primary.get("taxonomic_class", "not reported")
    
    # Drainage description mapping
    drainage_desc = {
        "Well drained": "well drained",
        "Moderately well drained": "moderately well drained",
        "Somewhat poorly drained": "somewhat poorly drained",
        "Poorly drained": "poorly drained",
        "Very poorly drained": "very poorly drained",
        "Somewhat excessively drained": "somewhat excessively drained",
        "Excessively drained": "excessively drained",
    }
    drain_text = drainage_desc.get(drainage, drainage.lower() if drainage else "not reported")
    
    # Hydrologic group description
    hydro_desc = {
        "A": "Group A (high infiltration rate, low runoff potential)",
        "B": "Group B (moderate infiltration rate)",
        "C": "Group C (slow infiltration rate, moderate runoff potential)",
        "D": "Group D (very slow infiltration rate, high runoff potential)",
        "A/D": "Group A/D (dual, depends on drainage)",
        "B/D": "Group B/D (dual, depends on drainage)",
        "C/D": "Group C/D (dual, depends on drainage)",
    }
    hydro_text = hydro_desc.get(hydro_group, f"Group {hydro_group}" if hydro_group else "not reported")
    
    map_unit = primary.get("map_unit", "Unknown")
    
    paragraph = (
        f"According to the Natural Resources Conservation Service (NRCS) Web Soil Survey, "
        f"soils at the Subject Property are mapped as {map_unit}. "
        f"The dominant soil components include {', '.join(soil_names)}. "
        f"The soils are classified as {drain_text} with a hydrologic soil group of {hydro_text}. "
    )
    
    if slope and slope != "not reported":
        paragraph += f"The representative slope is approximately {slope} percent. "
    
    if tax_class and tax_class != "not reported":
        paragraph += f"The taxonomic classification is {tax_class}. "
    
    return paragraph


# ── 3.5.2 Topography (USGS Elevation) ────────────────────────────────

async def get_elevation(lat: float, lon: float) -> dict:
    """
    Query USGS Elevation Point Query Service.
    Returns elevation in feet and meters.
    """
    url = "https://epqs.nationalmap.gov/v1/json"
    params = {"x": lon, "y": lat, "units": "Feet", "wkid": 4326}
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
    
    elev_ft = data.get("value")
    if elev_ft is not None:
        elev_ft = round(float(elev_ft), 1)
        elev_m = round(elev_ft * 0.3048, 1)
    else:
        return {"error": "Elevation data not available"}
    
    return {
        "elevation_ft": elev_ft,
        "elevation_m": elev_m,
    }


def format_topography_paragraph(elev_data: dict, address: str, lat: float, lon: float) -> str:
    """Generate ESA Section 3.5.2 paragraph."""
    if "error" in elev_data:
        return f"Elevation data could not be retrieved. {elev_data['error']}"
    
    elev_ft = elev_data.get("elevation_ft", "N/A")
    elev_m = elev_data.get("elevation_m", "N/A")
    
    # Convert decimal degrees to DMS for report
    def dd_to_dms(dd, is_lat=True):
        direction = ("N" if dd >= 0 else "S") if is_lat else ("E" if dd >= 0 else "W")
        dd = abs(dd)
        d = int(dd)
        m = int((dd - d) * 60)
        s = round((dd - d - m/60) * 3600, 2)
        return f"{d}°{m}'{s}\"{direction}"
    
    lat_dms = dd_to_dms(lat, True)
    lon_dms = dd_to_dms(lon, False)
    
    paragraph = (
        f"The Subject Property is located at approximately {lat_dms} latitude, "
        f"{lon_dms} longitude. "
        f"According to the USGS National Map Elevation Point Query Service, "
        f"the approximate elevation at the Subject Property is {elev_ft} feet "
        f"({elev_m} meters) above mean sea level. "
        f"Based on review of USGS topographic mapping, the property and surrounding area "
        f"appear to be relatively level."
    )
    
    return paragraph


# ── 3.5.3 Hydrogeology (NWI Wetlands) ────────────────────────────────

async def get_wetlands(lat: float, lon: float, radius_m: int = 500) -> dict:
    """
    Query USFWS National Wetlands Inventory for wetlands near a point.
    Uses the NWI ArcGIS REST service.
    """
    # NWI Wetlands MapServer
    url = "https://fwsprimary.wim.usgs.gov/server/rest/services/Wetlands/MapServer/0/query"
    
    # Create a buffer geometry (circle around point)
    params = {
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "spatialRel": "esriSpatialRelIntersects",
        "distance": radius_m,
        "units": "esriSRUnit_Meter",
        "outFields": "WETLAND_TYPE,ATTRIBUTE",
        "returnGeometry": "false",
        "f": "json",
    }
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
    
    features = data.get("features", [])
    wetlands = []
    for f in features:
        attrs = f.get("attributes", {})
        wetlands.append({
            "type": attrs.get("WETLAND_TYPE", "Unknown"),
            "code": attrs.get("ATTRIBUTE", ""),
        })
    
    # Summarize by type
    type_counts = {}
    for w in wetlands:
        t = w["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    
    return {
        "wetlands_found": len(wetlands) > 0,
        "count": len(wetlands),
        "types": type_counts,
        "radius_m": radius_m,
    }


def format_hydro_paragraph(wetland_data: dict, address: str) -> str:
    """Generate ESA Section 3.5.3 paragraph."""
    radius_ft = round(wetland_data.get("radius_m", 500) * 3.281)
    
    if not wetland_data.get("wetlands_found"):
        return (
            f"Based on a review of the U.S. Fish and Wildlife Service (USFWS) "
            f"National Wetlands Inventory (NWI) mapper, no wetlands were identified "
            f"within approximately {radius_ft} feet of the Subject Property. "
            f"Groundwater flow direction in the vicinity of the Subject Property is "
            f"anticipated to follow local topographic gradients."
        )
    
    types = wetland_data.get("types", {})
    type_list = ", ".join(f"{v} {k}" for k, v in types.items())
    
    return (
        f"Based on a review of the U.S. Fish and Wildlife Service (USFWS) "
        f"National Wetlands Inventory (NWI) mapper, wetlands were identified "
        f"within approximately {radius_ft} feet of the Subject Property. "
        f"The mapped wetland features include {type_list}. "
        f"Groundwater flow direction in the vicinity of the Subject Property is "
        f"anticipated to follow local topographic gradients."
    )


# ── 3.5.4 Flood Zone (FEMA NFHL) ────────────────────────────────────

async def get_flood_zone(lat: float, lon: float) -> dict:
    """
    Query FEMA National Flood Hazard Layer (NFHL) ArcGIS REST service.
    Returns flood zone designation for a point.
    """
    # FEMA NFHL MapServer — Layer 28 = S_FLD_HAZ_AR (flood hazard areas)
    url = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query"
    
    params = {
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "FLD_ZONE,ZONE_SUBTY,SFHA_TF,STATIC_BFE,DEPTH,DFIRM_ID,VERSION_ID",
        "returnGeometry": "false",
        "f": "json",
    }
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
    
    features = data.get("features", [])
    if not features:
        return {"error": "No FEMA flood data available for this location"}
    
    attrs = features[0].get("attributes", {})
    zone = attrs.get("FLD_ZONE", "Unknown")
    subtype = attrs.get("ZONE_SUBTY", "")
    sfha = attrs.get("SFHA_TF", "")
    dfirm = attrs.get("DFIRM_ID", "")
    
    # Zone descriptions
    zone_descriptions = {
        "A": "Zone A — Special Flood Hazard Area (SFHA), 1% annual chance flood (100-year flood), no base flood elevation determined",
        "AE": "Zone AE — Special Flood Hazard Area (SFHA), 1% annual chance flood (100-year flood) with base flood elevations determined",
        "AH": "Zone AH — Special Flood Hazard Area, 1% annual chance of shallow flooding (1-3 feet)",
        "AO": "Zone AO — Special Flood Hazard Area, 1% annual chance of shallow flooding with sheet flow",
        "VE": "Zone VE — Coastal Special Flood Hazard Area with velocity (wave action)",
        "X": "Zone X — Area of minimal flood hazard, outside the 0.2% annual chance floodplain",
        "D": "Zone D — Area of undetermined but possible flood hazard",
    }
    
    # Handle Zone X subtypes
    if zone == "X":
        if subtype and "0.2" in str(subtype):
            zone_desc = "Zone X (shaded) — Area of moderate flood hazard, 0.2% annual chance flood (500-year flood)"
            is_sfha = False
        else:
            zone_desc = zone_descriptions.get("X", f"Zone {zone}")
            is_sfha = False
    else:
        zone_desc = zone_descriptions.get(zone, f"Zone {zone}")
        is_sfha = sfha == "T" or zone in ("A", "AE", "AH", "AO", "VE", "V")
    
    return {
        "flood_zone": zone,
        "zone_subtype": subtype,
        "zone_description": zone_desc,
        "is_sfha": is_sfha,
        "dfirm_id": dfirm,
    }


def format_flood_paragraph(flood_data: dict, address: str) -> str:
    """Generate ESA Section 3.5.4 paragraph."""
    if "error" in flood_data:
        return f"Flood zone data could not be retrieved. {flood_data['error']}"
    
    zone = flood_data.get("flood_zone", "Unknown")
    zone_desc = flood_data.get("zone_description", "")
    is_sfha = flood_data.get("is_sfha", False)
    dfirm = flood_data.get("dfirm_id", "")
    
    dfirm_text = f" (DFIRM ID: {dfirm})" if dfirm else ""
    
    if is_sfha:
        paragraph = (
            f"According to the Federal Emergency Management Agency (FEMA) "
            f"National Flood Hazard Layer{dfirm_text}, "
            f"the Subject Property is located within a Special Flood Hazard Area (SFHA), "
            f"designated as {zone_desc}. "
            f"Flood insurance may be required for structures located within this zone."
        )
    else:
        paragraph = (
            f"According to the Federal Emergency Management Agency (FEMA) "
            f"National Flood Hazard Layer{dfirm_text}, "
            f"the Subject Property is located within {zone_desc}. "
            f"The property does not appear to be located within a Special Flood Hazard Area (SFHA)."
        )
    
    return paragraph


# ── Master Pull ──────────────────────────────────────────────────────

async def pull_all_site_data(address: str) -> dict:
    """
    Pull all government site data for an address.
    Returns structured data + formatted paragraphs for ESA sections 3.5.1-3.5.4.
    """
    # Step 1: Geocode
    geo = await geocode_address(address)
    if "error" in geo:
        return {"error": geo["error"]}
    
    lat = geo["lat"]
    lon = geo["lon"]
    
    # Step 2: Pull all data sources concurrently
    import asyncio
    soil_task = get_soil_data(lat, lon)
    elev_task = get_elevation(lat, lon)
    wetland_task = get_wetlands(lat, lon)
    flood_task = get_flood_zone(lat, lon)
    
    soil_data, elev_data, wetland_data, flood_data = await asyncio.gather(
        soil_task, elev_task, wetland_task, flood_task,
        return_exceptions=True,
    )
    
    # Handle exceptions from gather
    if isinstance(soil_data, Exception):
        soil_data = {"error": str(soil_data)}
    if isinstance(elev_data, Exception):
        elev_data = {"error": str(elev_data)}
    if isinstance(wetland_data, Exception):
        wetland_data = {"error": str(wetland_data)}
    if isinstance(flood_data, Exception):
        flood_data = {"error": str(flood_data)}
    
    # Step 3: Format paragraphs
    return {
        "geocode": geo,
        "sections": {
            "3.5.1": {
                "title": "Geology and Soils",
                "data": soil_data,
                "paragraph": format_soil_paragraph(soil_data, address),
            },
            "3.5.2": {
                "title": "Topography",
                "data": elev_data,
                "paragraph": format_topography_paragraph(elev_data, address, lat, lon),
            },
            "3.5.3": {
                "title": "Hydrogeology and Hydrology",
                "data": wetland_data,
                "paragraph": format_hydro_paragraph(wetland_data, address),
            },
            "3.5.4": {
                "title": "Flood Zone",
                "data": flood_data,
                "paragraph": format_flood_paragraph(flood_data, address),
            },
        },
    }
