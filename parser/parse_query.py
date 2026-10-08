"""
Week 2: natural language property query parser.

Turns a free-text search like "3 bed condos in Irvine under $1.5M with a pool"
into a structured filter object that Week 3 will turn into SQL against rets_property.

How it works:
1. Gemini reads the query and fills in a fixed JSON schema (PropertyFilters).
   Using a schema means Gemini can't invent extra fields or return free text.
2. validate() cleans up the result in plain Python, so bad values from the model
   (negative prices, made-up property types) never reach the database.

Usage:
    python parser/parse_query.py "3 bed condos in Irvine under $1.5M"
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from pydantic import BaseModel

# Exact L_Type_ values from rets_property. Check these against:
#   SELECT L_Type_, COUNT(*) FROM rets_property GROUP BY L_Type_;
# If the DB uses different strings, change them here. Everything else adapts.
PROPERTY_TYPES = [
    "SingleFamilyResidence",
    "Condominium",
    "Townhouse",
    "Duplex",
    "Triplex",
    "Quadruplex",
    "ManufacturedOnLand",
]

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")


class PropertyFilters(BaseModel):
    # No default values on purpose: the Gemini schema rejects them.
    # Every field is still nullable, so Gemini returns null for anything not mentioned.
    city: Optional[str]
    max_price: Optional[int]
    min_beds: Optional[int]
    min_baths: Optional[float]
    min_sqft: Optional[int]
    property_type: Optional[str]
    pool: Optional[bool]
    view: Optional[bool]
    max_hoa: Optional[int]


SYSTEM_PROMPT = f"""You convert real estate search requests into search filters.
Fill in only what the user actually asked for. Use null for everything else. Never guess.

Rules:
- city: the California city name with normal capitalization ("near Irvine" means Irvine).
- max_price: whole dollars. "1.5M" = 1500000, "900k" = 900000, "$650,000" = 650000.
  Words like under, below, max, up to, budget all mean max_price.
- min_beds, min_baths, min_sqft: a number like "3 bed" or "2.5 baths" or "1,800 sq ft" is a minimum.
- property_type: must be one of {PROPERTY_TYPES} or null.
  condo = Condominium. townhome/townhouse = Townhouse. house/single family = SingleFamilyResidence.
  duplex = Duplex. triplex = Triplex. fourplex/quadruplex = Quadruplex.
  manufactured/mobile home = ManufacturedOnLand. Land or lots are not in this data, so use null.
  Generic words (home, homes, listings, properties, places) = null.
- pool: true if they want a pool, false if they say no pool, null if not mentioned.
- view: true if they want a view of any kind, false if they say no view, null if not mentioned.
- max_hoa: monthly HOA fee limit in whole dollars.
"""


def load_env():
    """Read KEY=VALUE lines from the repo's .env so we don't need python-dotenv."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def validate(raw: dict) -> dict:
    """Clean up whatever the model returned before anything downstream uses it."""
    f = {field: raw.get(field) for field in PropertyFilters.model_fields}

    for key in ("max_price", "min_beds", "min_sqft", "max_hoa"):
        if f[key] is not None and f[key] <= 0:
            f[key] = None

    if f["min_baths"] is not None:
        if f["min_baths"] <= 0:
            f["min_baths"] = None
        else:
            f["min_baths"] = round(f["min_baths"] * 2) / 2  # snap to halves: 2.5, 3.0

    if f["property_type"] not in PROPERTY_TYPES:
        f["property_type"] = None

    if isinstance(f["city"], str):
        f["city"] = f["city"].strip() or None

    return f


def parse_property_query(query: str, client: Optional[genai.Client] = None) -> dict:
    if client is None:
        load_env()
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=PropertyFilters,
        temperature=0,  # same query, same answer
    )

    # Retry a few times so a free-tier rate limit doesn't kill the test run.
    for attempt in range(6):
        try:
            resp = client.models.generate_content(model=MODEL, contents=query, config=config)
            raw = resp.parsed.model_dump() if resp.parsed else json.loads(resp.text)
            return validate(raw)
        except Exception as e:
            if attempt == 5 or "503" not in str(e):
                raise
            print(f"Gemini overloaded, retrying in 20s...", file=sys.stderr)
            time.sleep(20)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python parser/parse_query.py "your search"')
        sys.exit(1)
    print(json.dumps(parse_property_query(" ".join(sys.argv[1:])), indent=2))
