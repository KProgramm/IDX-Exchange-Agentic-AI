"""
Runs the Week 2 parser against a set of test queries and reports pass/fail.

Several of these are cases the handbook's regex parser gets wrong:
"1,800 sq ft", "no pool", "Irvine condos", HOA limits, "2.25 million".

Usage:
    python tests/test_parser.py
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "parser"))
from parse_query import load_env, parse_property_query  # noqa: E402
from google import genai  # noqa: E402

EMPTY = {
    "city": None, "max_price": None, "min_beds": None, "min_baths": None,
    "min_sqft": None, "property_type": None, "pool": None, "view": None, "max_hoa": None,
}


def expect(**kwargs):
    return {**EMPTY, **kwargs}


CASES = [
    ("Show me 3-bedroom condos in Irvine under $1.5M with a pool",
     expect(city="Irvine", max_price=1500000, min_beds=3, property_type="Condominium", pool=True)),
    ("single family homes in Pasadena under 900k with at least 2.5 baths",
     expect(city="Pasadena", max_price=900000, min_baths=2.5, property_type="SingleFamilyResidence")),
    ("Irvine condos under 1M",
     expect(city="Irvine", max_price=1000000, property_type="Condominium")),
    ("4 bed house near San Diego with an ocean view",
     expect(city="San Diego", min_beds=4, property_type="SingleFamilyResidence", view=True)),
    ("homes in Newport Beach at least 1,800 sq ft",
     expect(city="Newport Beach", min_sqft=1800)),
    ("townhomes in Fremont with HOA under $400",
     expect(city="Fremont", property_type="Townhouse", max_hoa=400)),
    ("3 bed 2 bath in Sacramento, no pool, under $650,000",
     expect(city="Sacramento", max_price=650000, min_beds=3, min_baths=2, pool=False)),
    ("anything in Los Angeles under $2.25 million",
     expect(city="Los Angeles", max_price=2250000)),
    ("duplex in Riverside",
     expect(city="Riverside", property_type="Duplex")),
    ("land in Riverside",
     expect(city="Riverside")),
    ("2br condo in San Francisco with a view, HOA below 800 a month, max 1.2m",
     expect(city="San Francisco", max_price=1200000, min_beds=2, property_type="Condominium",
            view=True, max_hoa=800)),
    ("show me some listings",
     expect()),
    ("5 bedroom home with a pool and a view in Palo Alto, 3000 square feet minimum, budget 4.5M",
     expect(city="Palo Alto", max_price=4500000, min_beds=5, min_sqft=3000, pool=True, view=True)),
]


def main():
    load_env()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    passed = 0

    for i, (query, expected) in enumerate(CASES, 1):
        got = parse_property_query(query, client)
        time.sleep(13)
        diffs = {k: (expected[k], got[k]) for k in expected if expected[k] != got[k]}
        if diffs:
            print(f"FAIL {i:>2}: {query}")
            for k, (want, actual) in diffs.items():
                print(f"         {k}: expected {want!r}, got {actual!r}")
        else:
            passed += 1
            print(f"PASS {i:>2}: {query}")

    print(f"\n{passed}/{len(CASES)} passed")
    sys.exit(0 if passed == len(CASES) else 1)


if __name__ == "__main__":
    main()
