"""
Test script for map_retriever module.

Tests all three APIs with known coordinates.
"""

import asyncio
import sys
from pathlib import Path

from map_retriever import MapRetriever


async def test_location(name: str, lat: float, lon: float, output_dir: str):
    """Test a single location and print results."""
    print(f"\n{'='*60}")
    print(f"  {name} ({lat}, {lon})")
    print(f"{'='*60}\n")

    retriever = MapRetriever(output_dir=output_dir)
    try:
        result = await retriever.retrieve(lat, lon)
    finally:
        retriever.close()

    # Soil
    if result.soil:
        s = result.soil
        print("SOIL:")
        print(f"  Map Unit:         {s.map_unit_name}")
        print(f"  Component:        {s.component_name} ({s.component_pct}%)")
        print(f"  Drainage:         {s.drainage_class}")
        print(f"  Hydrologic Grp:   {s.hydrologic_group}")
        print(f"  Hydric:           {s.hydric_rating}")
        print(f"  Taxonomy:         {s.tax_class}")
        if s.corrosion_concrete:
            print(f"  Corrosion (conc): {s.corrosion_concrete}")
        if s.corrosion_steel:
            print(f"  Corrosion (steel):{s.corrosion_steel}")
        if s.map_image_path:
            print(f"  Map image:        {s.map_image_path}")
    else:
        print("SOIL: No data returned")

    print()

    # Wetlands
    if result.wetland:
        w = result.wetland
        print(f"WETLANDS: {'Present' if w.present else 'None found'}")
        if w.present:
            print(f"  Types:            {', '.join(w.wetland_types)}")
            print(f"  Feature count:    {len(w.features)}")
            for wf in w.features[:5]:
                acres_str = f", {wf.acres:.1f} ac" if wf.acres else ""
                print(f"    - {wf.attribute_code}: {wf.wetland_type} "
                      f"({wf.class_name}, {wf.water_regime}{acres_str})")
            if len(w.features) > 5:
                print(f"    ... and {len(w.features) - 5} more")
        if w.map_image_path:
            print(f"  Map image:        {w.map_image_path}")
    else:
        print("WETLANDS: No data returned")

    print()

    # Flood
    if result.flood:
        f = result.flood
        print("FLOOD:")
        print(f"  Zone:             {f.zone}")
        print(f"  SFHA:             {f.is_sfha}")
        print(f"  Risk:             {f.flood_risk_summary}")
        if f.zone_subtype:
            print(f"  Subtype:          {f.zone_subtype}")
        if f.base_flood_elevation_ft is not None:
            print(f"  Base elev (ft):   {f.base_flood_elevation_ft}")
        if f.dfirm_panel:
            print(f"  DFIRM panel:      {f.dfirm_panel}")
    else:
        print("FLOOD: No data returned")

    # Errors
    if result.errors:
        print(f"\nERRORS:")
        for api, err in result.errors.items():
            print(f"  {api}: {err}")

    print(f"\nRetrieved at: {result.retrieved_at}")
    print(f"Cached: {result.cached}")

    return result


async def main():
    # Test locations from the prompt
    locations = [
        ("San Diego (urban)", 32.7157, -117.1611),
        ("New Orleans (wetlands)", 29.9511, -90.0715),
        ("Houston (mixed zones)", 29.7604, -95.3698),
    ]

    output_dir = "./test_map_output"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for name, lat, lon in locations:
        loc_dir = f"{output_dir}/{name.split('(')[0].strip().lower().replace(' ', '_')}"
        await test_location(name, lat, lon, loc_dir)

    print(f"\n{'='*60}")
    print("  All tests complete. Check {output_dir}/ for map images.")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
