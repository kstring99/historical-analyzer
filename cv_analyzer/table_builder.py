"""Synthesize CV detections into ESA-formatted tables using a local LLM.

Takes structured CV output → local ollama LLM → Year/Issues/Observations tables
matching the format of the existing cloud-based historical analyzer.
"""

import re

from cv_analyzer.ollama_client import generate


SYSTEM_PROMPT = """You are a Phase I Environmental Site Assessment report writer.
You produce professional, concise observations for historical documentation tables.
Use ESA-standard language. Be factual and specific. No speculation.
Keep each observation to 1-2 sentences maximum."""


OBSERVATION_PROMPT = """Write a professional ESA observation for the {zone} based on these CV detections from a {year} aerial photograph.

Detections:
{detections}

Changes from prior period:
{changes}

Rules:
- Start with "The {zone_phrase}..."
- If no environmental concerns detected, describe general land use only
- If concerns detected, name EXACTLY what was found (e.g., "circular feature consistent with an underground storage tank")
- For Issues Noted: respond ONLY "Yes — [specific concern]" or "No"
- Keep observation to 1-2 sentences

Respond in EXACTLY this format:
OBSERVATION: [your observation]
ISSUES: [Yes — reason, or No]"""


ZONE_PHRASES = {
    "subject": "subject property appears to contain",
    "adjoining": "adjoining properties appear to include",
    "surrounding": "surrounding area appears to consist of",
    "unknown": "property appears to contain",
}


def build_tables(
    image_results: list[dict],
    change_results: list[dict],
    model: str = "llama3.2:3b",
    verbose: bool = False,
) -> dict:
    """Build ESA-formatted tables from CV analysis results.

    Returns:
    {
        "aerial": {
            "subject": [{"year_range": "1970 - 1990", "issues": "No", "observations": "..."}],
            "adjoining": [...],
            "surrounding": [...]
        }
    }
    """
    # Organize detections by year and zone
    year_zone_data = {}
    for img_res in image_results:
        year = img_res["year"]
        if year not in year_zone_data:
            year_zone_data[year] = {"subject": [], "adjoining": [], "surrounding": [], "unknown": []}
        for det in img_res.get("detections", []):
            zone = det.get("zone", "unknown")
            year_zone_data[year][zone].append(det)

    # Organize changes by target year
    year_changes = {}
    for ch_res in change_results:
        to_year = ch_res["to_year"]
        if to_year not in year_changes:
            year_changes[to_year] = []
        for ch in ch_res.get("changes", []):
            ch["from_year"] = ch_res["from_year"]
            year_changes[to_year].append(ch)

    years = sorted(year_zone_data.keys())

    # Generate observations per year per zone via LLM
    zone_rows = {"subject": [], "adjoining": [], "surrounding": []}

    for year in years:
        zones = year_zone_data[year]
        changes_for_year = year_changes.get(year, [])

        for zone in ("subject", "adjoining", "surrounding"):
            dets = zones.get(zone, []) + zones.get("unknown", [])
            zone_changes = [c for c in changes_for_year if c.get("zone", "unknown") in (zone, "unknown")]

            obs, issues = _generate_observation(
                year, zone, dets, zone_changes, model, verbose
            )

            zone_rows[zone].append({
                "year": year,
                "issues_noted": issues,
                "observations": obs,
            })

    # Group consecutive similar years into ranges
    tables = {}
    for zone in ("subject", "adjoining", "surrounding"):
        tables[zone] = _group_year_ranges(zone_rows[zone])

    return {"aerial": tables}


def _generate_observation(
    year: int,
    zone: str,
    detections: list[dict],
    changes: list[dict],
    model: str,
    verbose: bool,
) -> tuple[str, str]:
    """Generate a single observation + issues string via LLM."""

    # Format detections for the prompt
    if detections:
        det_lines = []
        for d in detections:
            det_lines.append(
                f"- {d['class'].replace('_', ' ')} (confidence {d['confidence']}): {d['description']}"
            )
        det_text = "\n".join(det_lines)
    else:
        det_text = "No specific features detected."

    if changes:
        ch_lines = []
        for c in changes:
            ch_lines.append(
                f"- {c['type'].replace('_', ' ')} (from {c.get('from_year', '?')}): "
                f"area={c['area_pct']}%, confidence={c['confidence']}"
            )
        ch_text = "\n".join(ch_lines)
    else:
        ch_text = "No changes from prior period (or first year in sequence)."

    prompt = OBSERVATION_PROMPT.format(
        zone=zone,
        zone_phrase=ZONE_PHRASES.get(zone, ZONE_PHRASES["unknown"]),
        year=year,
        detections=det_text,
        changes=ch_text,
    )

    if verbose:
        print(f"    [{year}/{zone}] Querying {model}...")

    try:
        response = generate(prompt, model=model, system=SYSTEM_PROMPT, temperature=0.2)
    except RuntimeError as e:
        if verbose:
            print(f"    [{year}/{zone}] LLM error: {e}")
        return _fallback_observation(detections, changes, zone), _fallback_issues(detections)

    # Parse response
    obs = ""
    issues = "No"
    for line in response.split("\n"):
        line = line.strip()
        if line.upper().startswith("OBSERVATION:"):
            obs = line.split(":", 1)[1].strip()
        elif line.upper().startswith("ISSUES:"):
            issues = line.split(":", 1)[1].strip()

    if not obs:
        obs = _fallback_observation(detections, changes, zone)
    if not issues:
        issues = _fallback_issues(detections)

    # Clean issues to table format
    if issues.lower().startswith("yes"):
        issues_clean = "Yes"
    else:
        issues_clean = "No"

    return obs, issues_clean


def _fallback_observation(detections: list, changes: list, zone: str) -> str:
    """Generate observation without LLM if it's unavailable."""
    if not detections and not changes:
        return f"The {zone} property appears developed with no environmental concerns noted."

    parts = []
    for d in detections[:3]:
        parts.append(d["description"])
    for c in changes[:2]:
        parts.append(f"{c['type'].replace('_', ' ')} detected")

    return f"The {zone} property contains: {'; '.join(parts)}."


def _fallback_issues(detections: list) -> str:
    concern_classes = {"circular_structure", "surface_staining", "vegetation_stress", "irregular_pond"}
    for d in detections:
        if d["class"] in concern_classes and d["confidence"] >= 0.5:
            return "Yes"
    return "No"


def _group_year_ranges(rows: list[dict]) -> list[dict]:
    """Group consecutive years with identical issues + similar observations."""
    if not rows:
        return []

    groups = [[rows[0]]]
    for row in rows[1:]:
        prev = groups[-1][-1]
        if row["issues_noted"] == prev["issues_noted"] and _obs_similar(row["observations"], prev["observations"]):
            groups[-1].append(row)
        else:
            groups.append([row])

    result = []
    for group in groups:
        if len(group) == 1:
            yr = str(group[0]["year"])
        else:
            yr = f"{group[0]['year']} - {group[-1]['year']}"
        result.append({
            "year_range": yr,
            "issues_noted": group[-1]["issues_noted"],
            "observations": group[-1]["observations"],
        })
    return result


def _obs_similar(a: str, b: str) -> bool:
    """Quick similarity check for grouping."""
    wa = set(re.findall(r"\b\w{4,}\b", a.lower()))
    wb = set(re.findall(r"\b\w{4,}\b", b.lower()))
    if not wa or not wb:
        return a == b
    overlap = len(wa & wb) / max(len(wa), len(wb))
    return overlap > 0.5


def format_tables_text(tables: dict) -> str:
    """Format tables as plain text for terminal/clipboard."""
    lines = []

    for doc_type, zones in tables.items():
        lines.append(f"{'=' * 80}")
        lines.append(f"  {doc_type.upper()} PHOTOGRAPH SUMMARY")
        lines.append(f"{'=' * 80}")

        for zone_name, rows in zones.items():
            zone_label = {
                "subject": "Subject Property",
                "adjoining": "Adjoining Properties",
                "surrounding": "Surrounding Properties",
            }.get(zone_name, zone_name.title())

            lines.append(f"\n--- {zone_label} ---")
            lines.append(f"{'Year':<20} {'Issues':<8} Observations")
            lines.append("-" * 80)

            for row in rows:
                yr = row["year_range"]
                issues = row["issues_noted"]
                obs = row["observations"]
                lines.append(f"{yr:<20} {issues:<8} {obs}")

            lines.append("")

    return "\n".join(lines)


def format_tables_html(tables: dict) -> str:
    """Format tables as HTML matching the existing app's ESA table output."""
    parts = []

    for doc_type, zones in tables.items():
        section_title = {
            "aerial": "AERIAL PHOTOGRAPH SUMMARY",
            "topo": "TOPOGRAPHIC MAP SUMMARY",
            "city_directory": "STREET DIRECTORY SUMMARY",
        }.get(doc_type, doc_type.upper())

        for zone_name, rows in zones.items():
            zone_label = {
                "subject": "Subject Property",
                "adjoining": "Adjoining Properties",
                "surrounding": "Surrounding Properties",
            }.get(zone_name, zone_name.title())

            obs_col = "Occupants" if doc_type == "city_directory" else "Observations"

            parts.append(f'<p style="text-align:center;font-weight:bold;margin-bottom:4px">'
                         f'{section_title} - {zone_label}</p>')
            parts.append('<table border="1" cellpadding="8" cellspacing="0" '
                         'style="border-collapse:collapse;width:100%">')
            parts.append(f"<thead><tr><th>Year</th><th>Issues Noted</th>"
                         f"<th>{obs_col}</th></tr></thead><tbody>")

            for row in rows:
                parts.append(
                    f"<tr>"
                    f"<td style='white-space:nowrap;vertical-align:top'>{row['year_range']}</td>"
                    f"<td style='text-align:center;vertical-align:top'>{row['issues_noted']}</td>"
                    f"<td>{row['observations']}</td>"
                    f"</tr>"
                )

            parts.append("</tbody></table><br>")

    return "\n".join(parts)
