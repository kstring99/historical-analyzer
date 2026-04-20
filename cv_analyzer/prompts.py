"""ESA-standard prompts for historical documentation analysis.

Written in the voice of the report author. No CV jargon, no pixel dimensions,
no "low-saturation areas" — describe what a reader would see on the image.
"""

# ---- Shared instructions used in every three-zone prompt ----

THREE_ZONE_RULES = """RULES FOR PROFESSIONAL ESA LANGUAGE:
- Write in the voice of a Phase I ESA report author. Clear, factual, present tense.
- Always capitalize "Subject Property" as a proper noun wherever it appears in your
  observations (e.g., "The Subject Property appears to..."). The adjoining and
  surrounding descriptors remain lowercase.
- Never mention pixel counts, dimensions in "px", saturation, contrast, brightness, or
  any words a computer vision algorithm would use. Describe what a human reader sees.
- Never invent numeric confidence values. Either a feature is present or it isn't.
- Keep each zone description to 1–2 sentences.
- Use consistent terminology across years so similar land use groups into year ranges.

ISSUES NOTED RULES — only mark "Yes" when you can clearly identify a specific
environmental concern (tanks, fuel canopies, drums, staining, distressed vegetation,
industrial equipment, mining/disposal/treatment symbols on a topo, gas station in
a city directory, etc.). Normal buildings, parking lots, residential, commercial,
undeveloped, agricultural → "No". When "Yes", state the business or feature and
what makes it a concern. Do not flag ambiguous features (a roundabout is not a tank,
a baseball diamond is not a drum cluster).
"""


# ---- Cover page ----

COVER_PAGE_PROMPT = """This is the cover page of an ERIS historical documentation package.

Extract the following metadata if present, one per line as "Key: Value":
- Property Address
- Coordinates (lat/long)
- Project Property Name
- Project Number
- Order Number

Omit any field you cannot find. No other commentary."""


# ---- Aerial ----

AERIAL_PROMPT = """You are a Phase I Environmental Site Assessment author describing a historical aerial photograph.

A green rectangle on the image marks the subject property boundary.

{cv_hints}

Describe THREE zones exactly as a human reviewer would, then flag any environmental concerns.

1. SUBJECT PROPERTY — land use inside the green rectangle.
2. ADJOINING PROPERTIES — the properties immediately sharing a boundary with the
   green rectangle. Use directional references (north, south, east, west).
3. SURROUNDING PROPERTIES — the wider visible area beyond the adjoining properties.

Extract the YEAR from the footer bar of the photograph.

""" + THREE_ZONE_RULES + """

Format your response EXACTLY as:
YEAR: [year]
SUBJECT: [description starting with "The Subject Property appears to..."]
SUBJECT_ISSUES: [No — OR — Yes, brief description of what you see]
ADJOINING: [description with directional references]
ADJOINING_ISSUES: [No — OR — Yes, brief description]
SURROUNDING: [description]
SURROUNDING_ISSUES: [No — OR — Yes, brief description]"""


# ---- Topographic map ----

TOPO_PROMPT = """You are a Phase I Environmental Site Assessment author describing a USGS topographic map.

The subject property is located at approximately {coordinates}. {context}

Describe THREE zones as they appear on the map, then flag any environmental concerns.

1. SUBJECT PROPERTY — what the map shows at the property location (terrain, structures
   shown as squares, roads, water features).
2. ADJOINING PROPERTIES — immediate vicinity of the property location, with directional
   references.
3. SURROUNDING PROPERTIES — wider area visible on the map.

Environmental concerns on topographic maps include: mine / quarry symbols, tailings,
landfill or disposal-site annotations, industrial facility symbols, pipelines, tank farms,
sewage or water treatment plants. Normal roads, buildings, and residential areas are
not concerns.

Extract the YEAR from the footer or margin.

""" + THREE_ZONE_RULES + """

Format your response EXACTLY as:
YEAR: [year]
SUBJECT: [description starting with "The Subject Property appears to..."]
SUBJECT_ISSUES: [No — OR — Yes, specific map feature]
ADJOINING: [description with directional references]
ADJOINING_ISSUES: [No — OR — Yes, specific map feature]
SURROUNDING: [description]
SURROUNDING_ISSUES: [No — OR — Yes, specific map feature]"""


# ---- Fire insurance map (Sanborn / FIM) ----

FIM_PROMPT = """You are a Phase I Environmental Site Assessment author describing a historical fire insurance map (Sanborn or equivalent).

The Subject Property is marked by a RED rectangle or red outline on the map.

The Subject Property is located at approximately {coordinates}. {context}

Fire insurance maps show building footprints, construction material (brick, stone, frame, iron-clad), occupancy labels (dwelling, store, stable, shop), and notable features (fuel tanks, oil sheds, pumps, coal bins, kilns, ice houses). They are one of the most detailed historical sources for environmental concerns on urban and semi-urban sites.

Describe THREE zones, then flag any environmental concerns.

1. SUBJECT PROPERTY — what the map shows inside the red boundary. Building footprints, occupancy labels, construction material, any tanks, pumps, or stored materials. Empty lots and "Vacant" labels are informative too.
2. ADJOINING PROPERTIES — the properties immediately across the street or sharing a boundary with the red rectangle. Use directional references (north, south, east, west).
3. SURROUNDING PROPERTIES — the wider map context beyond adjoining parcels.

Environmental concerns on fire insurance maps include:
- Fuel / oil / gasoline tanks or pumps (labels like "Gasoline", "Oil", "Fuel", "Tks", "UST", "Pump")
- Service stations, auto repair/paint/body shops, auto wrecking/salvage yards
- Dry cleaners or laundries using solvents ("Dry Cleaner", "Cleaners & Dyers", "Naphtha")
- Machine shops, metalworking, plating, foundries, blacksmiths, welders
- Coal bins, coal yards, coke ovens
- Chemical storage, paint stores, hardware with chemical sales
- Printing / lithography (often using solvents and inks)
- Ice houses, tanneries, slaughterhouses, creameries
- Lumber yards with treatment/preservation operations
- Manufactured gas plants (MGPs) — labelled "Gas Works", "Gas Plant", "Gasometer"

A plain "Dwelling", "Store", or "Office" is NOT a concern.

Extract the YEAR from the map header or margin. If a range is shown (e.g., "Corrected to 1942"), take the most recent year.

""" + THREE_ZONE_RULES + """

Format your response EXACTLY as:
YEAR: [year]
SUBJECT: [description starting with "The Subject Property appears to..."]
SUBJECT_ISSUES: [No — OR — Yes, specific map feature or label]
ADJOINING: [description with directional references]
ADJOINING_ISSUES: [No — OR — Yes, specific map feature or label]
SURROUNDING: [description]
SURROUNDING_ISSUES: [No — OR — Yes, specific map feature or label]"""


# ---- City directory ----

CITY_DIR_EXTRACT_PROMPT = """You are reading a page from a historical city directory for a Phase I ESA.

Extract every directory entry visible on this page. For each entry include:
- The year (from the page or book header)
- The street name (from the section header)
- The street number
- The occupant or business name
- A short use classification (residence, retail, auto repair, gasoline station, dry cleaner, etc.)
- Whether the use is environmentally concerning (gasoline/filling/fuel, auto repair
  or body, auto salvage, dry cleaner using solvents, chemical storage or distribution,
  printing/lithography, metalworking/plating, paint/hardware with chemical sales,
  pest control, photo processing, manufacturing). Residential, retail, restaurants,
  schools, offices, churches are NOT concerns.

Format your response as:
YEAR: [year]
STREET: [street name]
ENTRIES:
[number] | [occupant] | [classification] | [ENV_FLAG if concerning, else NONE]
...
"""


CITY_DIR_SYNTHESIS_PROMPT = """You are the author of the city directory section of a Phase I ESA.

Subject Property address: {subject_address}
Adjoining property addresses (with directions): {adjoining_addresses}

Raw directory data extracted from the pages:
{raw_entries}

For EACH year that has data, produce entries for three zones.

1. SUBJECT PROPERTY — the listing at the Subject Property address. If there is no
   listing for a year, write "No listing identified for the Subject Property address."
2. ADJOINING PROPERTIES — listings at the adjoining addresses. Lead with the
   directional reference when provided (e.g., "The adjoining property to the north
   at 6510 Wentworth Springs Rd was listed as..."). If no listings, write
   "No listings identified for adjoining property addresses."
3. SURROUNDING — other nearby listings that are environmentally relevant.

Environmental concerns include gasoline stations, auto repair/body/salvage, dry
cleaners using solvents, chemical manufacturing or storage, printing/lithography,
machine shops, metalworking/plating, paint/hardware with chemical sales, pest
control, photo processing. Residential, retail, restaurants, schools, offices,
churches are NOT concerns.

""" + THREE_ZONE_RULES + """

Format EACH year EXACTLY as:
YEAR: [year]
SUBJECT: [listing or "No listing identified for the Subject Property address."]
SUBJECT_ISSUES: [No — OR — Yes, specific business and use]
ADJOINING: [listings or "No listings identified for adjoining property addresses."]
ADJOINING_ISSUES: [No — OR — Yes, specific business and use]
SURROUNDING: [relevant nearby listings or "No environmentally significant listings identified in the surrounding area."]
SURROUNDING_ISSUES: [No — OR — Yes, specific business and use]
"""


# ---- Year-range consolidation ----

CONSOLIDATION_PROMPT = """You are consolidating per-year observations from a {doc_label} review into year-range groupings for a Phase I ESA table. You are consolidating the {zone_label} column only.

Per-year observations (year | issues | observation):
{entries}

RULES
1. GROUP consecutive years that describe the SAME land use with the SAME issue status. Minor wording differences across years ("agricultural field" vs "agricultural land") do NOT warrant a new row — same land use → same group.
2. BREAK a group when:
   - The land use materially changes (new structure built, demolition, cleared, newly developed, reverted to vacant, road added, etc.).
   - A NEW environmental concern appears, or an existing one clearly goes away.
3. When a concern flickers across adjacent years (e.g., 2004 Yes, 2006 No, 2008 Yes on what is obviously the same agricultural field), treat the whole span as ONE row with Issues = Yes and note in the observation that the concern is intermittent, naming the specific years it was observed. Do NOT produce alternating single-year rows for a field that hasn't changed.
4. For each group, write ONE observation (1-2 sentences) that captures the shared land use. If there was a transition within the range, say so briefly ("…until 1994, when a small building complex appeared along the western boundary.").
5. If the group contains any Yes years, the consolidated row is Yes and the observation must name the specific concern.
6. Professional ESA voice. No pixel measurements, no CV jargon, no confidence numbers, no model self-reference.
7. Preserve the SORT ORDER. Output earliest range first.

OUTPUT FORMAT — one line per range, pipe-separated, nothing else:
YEAR_RANGE | Yes|No | observation

Example output:
1937 - 1994 | No | The subject property appears to be undeveloped agricultural land with visible crop rows throughout the observation period.
2004 - 2025 | Yes | The subject property continues to appear as active agricultural land. Possible distressed vegetation and soil discoloration are intermittently visible within the field (2004, 2008, 2010-2015, 2021, and 2025)."""


# ---- Cross-reference summary ----

SUMMARY_PROMPT = """You are writing the summary paragraph for the Historical Documentation
section of a Phase I Environmental Site Assessment.

Sources reviewed:
{findings}

Write a concise, professional summary paragraph that:

1. States the date range of each source reviewed (e.g., "Historical sources reviewed
   included aerial photographs from 1946 through 2022, topographic maps from 1949
   through 2022, and city directory listings from 1995 through 2020.").

2. Summarizes the general development history of the subject property and
   surrounding area.

3. Cross-references findings between document types:
   - If a city directory lists an environmental concern at an address AND an aerial
     from the same period shows corroborating visual evidence in the same direction,
     say both sources corroborate the concern.
   - If only one source flags a concern, say so and note that the other source did
     not show visible indicators.

4. Clearly states whether environmental concerns were identified. If yes, state:
   WHAT the concern is, WHERE (subject / adjoining / surrounding), WHICH source(s)
   identified it, and WHAT time period.

5. If no environmental concerns were identified across any source, end with:
   "No environmental concerns were identified in the historical documentation reviewed."

Professional ESA tone. Factual. No speculation. No pixel or CV jargon.
"""
