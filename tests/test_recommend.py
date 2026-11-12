"""
Week 7 tests for the recommendation engine. Zero Gemini calls. Almost everything runs on a small
synthetic index and a fake database; the last section runs read-only queries against MySQL and is
skipped if it's unreachable.

Usage:
    python tests/test_recommend.py
"""

import math
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from recommend import recommend as R  # noqa: E402
from search.search_listings import PUBLIC_COLUMNS  # noqa: E402

failures = []


def check(name, condition):
    print(("PASS" if condition else "FAIL"), name)
    if not condition:
        failures.append(name)


WINDOW = ("2026-03-17", "2026-09-17", 6.0, "2026-03-18")  # (start exclusive, end, months, data_start)


def unit_vec(cos_to_axis):
    """A unit vector whose cosine with [1, 0, 0] is exactly `cos_to_axis`."""
    return np.array([cos_to_axis, math.sqrt(1 - cos_to_axis**2), 0.0], dtype=np.float32)


def listing(i, cos=None, **kw):
    row = {c: None for c in PUBLIC_COLUMNS}
    row.update(
        id=i, L_Address=f"{i} Oak St", L_City="Irvine", L_State="CA", L_Zip="92618", L_Type_="Condominium",
        L_SystemPrice=500_000, L_Keyword2=2, LM_Dec_3=2, LM_Int2_3=1000, ParcelNumber=f"P{i}", _cos=cos,
    )
    row.update(kw)
    return row


def build_index(rows):
    """Fake (_load result): listings DataFrame and vectors, where row _cos sets the cosine to listing 1."""
    vectors, df_rows = [], []
    for r in rows:
        r = dict(r)
        cos = r.pop("_cos")
        if cos is None:
            r["_vec"] = -1
        else:
            r["_vec"] = len(vectors)
            vectors.append(unit_vec(cos))
        df_rows.append(r)
    df = pd.DataFrame(df_rows)
    for col in ("L_SystemPrice", "L_Keyword2", "LM_Dec_3", "LM_Int2_3"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["_addr"] = df["L_Address"].map(R._norm_addr)
    df["_city"] = df["L_City"].fillna("").str.strip().str.lower()
    df["_parcel"] = df["ParcelNumber"].map(R._norm_parcel)
    return df, np.array(vectors, dtype=np.float32)


def comp_rows(n, price=500_000, sqft=1000):
    return [{"ClosePrice": price, "LivingArea": sqft} for _ in range(n)]


def run(rows, listing_id=1, comps=None, **kw):
    """recommend() against a synthetic index; every comp query answers with `comps` (default: 10 comps)."""
    calls = []

    def fake_query(sql, params=None):
        calls.append((sql, list(params or [])))
        return comp_rows(10) if comps is None else comps

    with mock.patch.object(R, "_load", return_value=build_index(rows)), \
         mock.patch.object(R, "_data_window", return_value=WINDOW), \
         mock.patch.object(R, "query", fake_query):
        res = R.recommend(listing_id, **kw)
    return res, calls


# ---------- scoring ----------
print("-- scoring --")
target = listing(1, cos=1.0)
res, _ = run([target, listing(2, cos=1.0)])
c = res["recommendations"][0]["components"]
check("identical listing scores full points on every component",
      c == {"price": 20.0, "beds": 15.0, "city": 15.0, "sqft": 10.0, "semantic": 40.0}
      and res["recommendations"][0]["score"] == 100.0)

# Problem 1: the same $50k gap hurts a $400k target far more than a $4M one.
cheap, _ = run([listing(1, cos=1.0, L_SystemPrice=400_000), listing(2, cos=1.0, L_SystemPrice=450_000)])
rich, _ = run([listing(1, cos=1.0, L_SystemPrice=4_000_000), listing(2, cos=1.0, L_SystemPrice=4_050_000)])
check("price gap is relative: $50k off a $400k target scores less than $50k off a $4M target",
      cheap["recommendations"][0]["components"]["price"] < rich["recommendations"][0]["components"]["price"])
far, _ = run([listing(1, cos=1.0), listing(2, cos=1.0, L_SystemPrice=900_000)])
check("price 80% apart scores 0", far["recommendations"][0]["components"]["price"] == 0.0)

# Problem 2: graded beds.
res, _ = run([listing(1, cos=1.0, L_Keyword2=3)] + [listing(10 + b, cos=1.0, L_Keyword2=b) for b in (3, 4, 5, 6, 9)], top_k=10)
beds = {r["id"] - 10: r["components"]["beds"] for r in res["recommendations"]}
check("beds: 0/1/2/3+ apart score 15/10/5/0 (3 vs 9 no longer equals 3 vs 4)",
      beds[3] == 15 and beds[4] == 10 and beds[5] == 5 and beds[6] == 0 and beds[9] == 0)

# Problem 4: semantic rescale at the fixed bounds.
res, _ = run([listing(1, cos=1.0), listing(2, cos=0.30), listing(3, cos=0.60), listing(4, cos=0.95)], top_k=10)
sem = {r["id"]: r["components"]["semantic"] for r in res["recommendations"]}
check("semantic: cos 0.30 -> 0, 0.60 -> 20, 0.95 -> capped at 40", sem[2] == 0.0 and sem[3] == 20.0 and sem[4] == 40.0)

# ---------- type filter ----------
print("-- property type --")
res, _ = run([listing(1, cos=1.0)] + [listing(i, cos=0.9, L_Type_="SingleFamilyResidence") for i in (2, 3)] + [listing(4, cos=0.5)])
check("condo target is never offered a single-family home", [r["id"] for r in res["recommendations"]] == [4])
check("candidates_scored counts only the type-matched pool", res["candidates_scored"] == 1)

res, _ = run([listing(1, cos=1.0, L_Type_=None), listing(2, cos=0.9, L_Type_="SingleFamilyResidence"), listing(3, cos=0.8)])
check("NULL-type target: no type filter", {r["id"] for r in res["recommendations"]} == {2, 3})
check("NULL-type target: flagged", "type unknown: not filtered by type" in res["flags"])
check("NULL-type target: comp query has no subtype clause",
      "PropertySubType" not in run([listing(1, cos=1.0, L_Type_=None), listing(2, cos=0.9)])[1][0][0])
check("typed target: comp query filters PropertySubType",
      "PropertySubType = %s" in run([listing(1, cos=1.0), listing(2, cos=0.9)])[1][0][0])

# ---------- duplicates ----------
print("-- duplicates --")
rows = [
    listing(1, cos=1.0, L_Address="100 Main St 5", ParcelNumber="A1"),
    listing(2, cos=0.995, L_Address="100 Main St #5", ParcelNumber="A1"),  # relist of the same unit
    listing(3, cos=0.995, L_Address="100 Main St 6", ParcelNumber="A2"),  # same building, other unit
    listing(4, cos=0.995, L_Address="100 Main St 5", ParcelNumber="A9"),  # same string, different parcel
    listing(5, cos=0.90, L_Address="100 Main St 5", ParcelNumber="A1"),  # same property but below the 0.99 check
]
res, _ = run(rows, top_k=10)
ids = {r["id"] for r in res["recommendations"]}
check("same normalised address + city is excluded (#2)", 2 not in ids and res["excluded_same_property"] == 1)
check("same building, different unit is kept (#3)", 3 in ids)
check("same address string but conflicting parcel is kept (#4)", 4 in ids)
check("same-property check only applies at cosine >= 0.99 (#5 kept)", 5 in ids)
check("_same_property: different city is a different property",
      not R._same_property(np.array(["1 a st"]), np.array(["x"]), np.array([""]), "1 a st", "y", "")[0])

# ---------- guards ----------
print("-- guards --")
res, _ = run([listing(1, cos=1.0, L_SystemPrice=100), listing(2, cos=0.8)])
check("placeholder target price: price not scored, flagged",
      res["recommendations"][0]["components"]["price"] == 0.0 and any("placeholder" in f for f in res["flags"]))
check("placeholder target price: no delta, no crash", res["target"]["comps"]["status"] == "no_price" and res["target"]["comps"]["delta_pct"] is None)

res, _ = run([listing(1, cos=None), listing(2, cos=0.8), listing(3, cos=None)])
check("target with no vector: structured only, flagged, scaled to 100",
      any("no remarks vector" in f for f in res["flags"]) and res["recommendations"][0]["score"] == 100.0
      and all(r["cosine"] is None and r["components"]["semantic"] == 0.0 for r in res["recommendations"]))
try:
    with mock.patch.object(R, "_load", return_value=build_index([listing(1, cos=None), listing(2, cos=0.5)])):
        R.recommend(1, mode="semantic")
    check("semantic mode refuses a target with no vector", False)
except ValueError:
    check("semantic mode refuses a target with no vector", True)

res, _ = run([listing(1, cos=1.0), listing(2, cos=None), listing(3, cos=0.47)])
neutral = {r["id"]: r for r in res["recommendations"]}
check("candidate with no vector gets the neutral semantic score (same as cos 0.47) and a note",
      neutral[2]["components"]["semantic"] == neutral[3]["components"]["semantic"]
      and "no remarks vector: neutral semantic score" in neutral[2]["flags"])

res, _ = run([listing(1, cos=1.0, LM_Int2_3=None), listing(2, cos=0.8)])
check("target sqft missing: sqft not scored, flagged",
      res["recommendations"][0]["components"]["sqft"] == 0.0 and any("sqft missing" in f for f in res["flags"]))
try:
    run([listing(1, cos=1.0)], listing_id=999)
    check("unknown listing id raises ValueError", False)
except ValueError:
    check("unknown listing id raises ValueError", True)
res, _ = run([listing(1, cos=1.0)] + [listing(i, cos=0.9) for i in range(2, 9)], top_k=3)
check("top_k limits results; target never recommends itself", len(res["recommendations"]) == 3 and 1 not in {r["id"] for r in res["recommendations"]})

res, _ = run([listing(1, cos=1.0), listing(2, cos=0.9, L_Address=None, L_Zip=None)])
check("NULL address renders as city/state, not 'None'", "None" not in res["recommendations"][0]["address"])

res, _ = run([listing(1, cos=1.0), listing(2, cos=0.2, L_Type_="Cabin"), listing(3, cos=0.9, L_City="Reno")], mode="semantic")
check("pure semantic mode ranks by raw cosine with no type filter", [r["id"] for r in res["recommendations"]] == [3, 2])

# ---------- comp validation ----------
print("-- comps --")
with mock.patch.object(R, "query", return_value=[]):
    out = R.validate_with_comps("Irvine", "Condominium", 1000, 500_000, WINDOW)
check("zero comps: insufficient, no delta, no ZeroDivisionError",
      out["status"] == "insufficient" and out["delta_pct"] is None and out["comp_count"] == 0)
with mock.patch.object(R, "query", return_value=comp_rows(R.MIN_COMPS - 1)):
    out = R.validate_with_comps("Irvine", "Condominium", 1000, 500_000, WINDOW)
check("4 comps is below the minimum of 5", out["status"] == "insufficient" and out["delta_pct"] is None)
with mock.patch.object(R, "query", return_value=comp_rows(R.MIN_COMPS)):
    out = R.validate_with_comps("Irvine", "Condominium", 1000, 550_000, WINDOW)
check("5 comps is enough: median $/sqft x sqft, delta +10%",
      out["status"] == "ok" and out["comp_price"] == 500_000 and out["delta_pct"] == 10.0 and out["comp_count"] == 5)
skewed = comp_rows(5) + [{"ClosePrice": 9_000_000, "LivingArea": 1000}]
with mock.patch.object(R, "query", return_value=skewed):
    out = R.validate_with_comps("Irvine", "Condominium", 1000, 500_000, WINDOW)
check("median ignores one outlier sale", out["comp_price"] == 500_000)
for price, expected in [(500_000, "in line with comps"), (524_000, "in line with comps"), (540_000, "above comps"),
                        (580_000, "well above comps"), (460_000, "below comps"), (400_000, "well below comps")]:
    with mock.patch.object(R, "query", return_value=comp_rows(10)):
        check(f"verdict for list ${price:,}: {expected}",
              R.validate_with_comps("Irvine", "Condominium", 1000, price, WINDOW)["verdict"] == expected)
for sqft in (None, 0, float("nan")):
    check(f"sqft {sqft}: insufficient without querying",
          R.validate_with_comps("Irvine", "Condominium", sqft, 500_000, WINDOW)["status"] == "insufficient")
check("no sold data at all: insufficient", R.validate_with_comps("Irvine", "Condominium", 1000, 500_000, None)["status"] == "insufficient")
check("no city: insufficient", R.validate_with_comps("", "Condominium", 1000, 500_000, WINDOW)["status"] == "insufficient")

captured = []
with mock.patch.object(R, "query", lambda sql, params=None: captured.append((sql, params)) or comp_rows(10)):
    out = R.validate_with_comps("Irvine", "Condominium", 1000, 500_000, WINDOW)
sql, params = captured[0]
check("comp window comes from the data (exclusive start, end), not CURDATE()",
      "CURDATE" not in sql and params[3:5] == ["2026-03-17", "2026-09-17"] and out["window_start"] == "2026-03-18" and out["window_end"] == "2026-09-17")
check("comp query is parameterised with city, +/-20% sqft band, subtype",
      params[0] == "Irvine" and params[1:3] == [800.0, 1200.0] and params[5] == "Condominium" and "%s" in sql and "Irvine" not in sql)
res, _ = run([listing(1, cos=1.0), listing(2, cos=0.9)], comps=comp_rows(2))
check("insufficient comps are counted across target + recs", res["comps_insufficient"] == 2 and res["comps_total"] == 2)

# ---------- live (read-only) ----------
print("-- live --")
try:
    R._load.cache_clear()
    df, _ = R._load()
    res = R.recommend(368235)
except Exception as e:  # DB or index unreachable
    print("SKIP live checks:", type(e).__name__, str(e)[:80])
else:
    recs = res["recommendations"]
    check("live: 5 recs, unique, not the target", len(recs) == 5 and len({r["id"] for r in recs}) == 5 and 368235 not in {r["id"] for r in recs})
    check("live: every rec has the target's type", all(r["type"] == res["target"]["type"] for r in recs))
    check("live: scores descending and within 0-100", [r["score"] for r in recs] == sorted((r["score"] for r in recs), reverse=True) and 0 <= recs[-1]["score"] <= 100)
    check("live: comp window anchored on the data, not 2072", res["comp_window"]["end"] == "2026-09-17")
    check("live: no 'None' in addresses", all("None" not in r["address"] for r in recs))
    nv = df[(df["_vec"] < 0) & df["L_Type_"].notna()]
    if len(nv):
        r2 = R.recommend(int(nv.iloc[0]["id"]))
        check("live: target with no vector still returns recs, flagged", len(r2["recommendations"]) > 0 and any("no remarks vector" in f for f in r2["flags"]))

print()
print(f"{len(failures)} failure(s)" if failures else "All checks passed")
sys.exit(1 if failures else 0)
