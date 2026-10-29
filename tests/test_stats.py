"""
Week 5 tests for the market stats. Zero Gemini calls. Most checks mock the database;
the last section runs a few read-only queries against MySQL and is skipped if it's unreachable.

Usage:
    python tests/test_stats.py
"""

import json
import sys
from decimal import Decimal
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from market import stats  # noqa: E402

failures = []


def check(name, condition):
    print(("PASS" if condition else "FAIL"), name)
    if not condition:
        failures.append(name)


class FakeDB:
    """Stands in for db.connection.query. Records every call and answers by SQL content."""

    def __init__(self, sold_rows=(), trend_rows=(), active=0, sold_count=0, window=("2026-03-18", "2026-09-17")):
        self.calls = []
        self.sold_rows, self.trend_rows = list(sold_rows), list(trend_rows)
        self.active, self.sold_count, self.window = active, sold_count, window

    def __call__(self, sql, params=None):
        self.calls.append((sql, list(params or [])))
        if "MIN(CloseDate)" in sql:
            first, last = self.window
            return [{"first_sale": first, "last_sale": last}] if first else [{"first_sale": None, "last_sale": None}]
        if "FROM rets_property" in sql:
            return [{"n": self.active}]
        if "COUNT(*) AS n" in sql:
            return [{"n": self.sold_count}]
        if "GROUP BY" in sql:
            return self.trend_rows
        return self.sold_rows


def sale(price, lst, sqft, dom):
    return {"ClosePrice": price, "ListPrice": lst, "LivingArea": sqft, "DaysOnMarket": dom}


def run(fn, fake, *args, **kw):
    with mock.patch.object(stats, "query", fake):
        return fn(*args, **kw)


# 1. city_summary math. Prices are Decimals on purpose: MySQL can hand those back.
rows = [
    sale(Decimal("1000000"), Decimal("1000000"), 2000, 10),   # $500/sqft, ratio 100%
    sale(Decimal("2000000"), Decimal("2500000"), 1000, 30),   # $2000/sqft, ratio 80%
    sale(Decimal("300000"), Decimal("300000"), 0, 20),        # no sqft: excluded from $/sqft only
    sale(Decimal("700000"), Decimal("0"), 1400, None),        # no list price, no DOM
]
fake = FakeDB(sold_rows=rows)
s = run(stats.city_summary, fake, "Irvine")
check("summary has_data", s["has_data"] is True)
check("sold_count counts every valid sale", s["sold_count"] == 4)
check("avg close price", s["avg_close_price"] == 1000000)
check("median is the middle of the sorted prices, not the mean", s["median_close_price"] == 850000)
check("price/sqft skips LivingArea = 0", s["price_per_sqft_sample"] == 3 and s["avg_price_per_sqft"] == 1000)
check("avg DOM skips NULL", s["avg_dom"] == 20.0)
check("list-to-close skips ListPrice = 0", s["list_to_close_pct"] == 93.3)
check("no Decimals in result", all(not isinstance(v, Decimal) for v in s.values()))
check("result is JSON-serializable", bool(json.dumps(s)))

# 2. Median with an odd count, and different from the mean.
odd = FakeDB(sold_rows=[sale(100, 100, 1, 1), sale(200, 200, 1, 1), sale(1000, 1000, 1, 1)])
s = run(stats.city_summary, odd, "Irvine")
check("odd-count median", s["median_close_price"] == 200 and s["avg_close_price"] == 433)

# 3. SQL safety: every query is parameterized and filtered.
sql, params = fake.calls[-1]
check("city is a %s parameter, never inlined", "Irvine" not in sql and "Irvine" in params)
check("summary filters PropertyType", "PropertyType = 'Residential'" in sql)
check("summary filters PropertySubType", "PropertySubType = %s" in sql and "SingleFamilyResidence" in params)
check("summary filters ClosePrice > 0", "ClosePrice > 0" in sql)
check("default subtype is SFR, condos are not mixed in", stats.DEFAULT_SUBTYPE == "SingleFamilyResidence")

# 4. Injection attempt stays a parameter.
evil = "Irvine'; DROP TABLE california_sold; --"
fake_evil = FakeDB()
run(stats.city_summary, fake_evil, evil)
check("hostile city never reaches the SQL text",
      all("DROP" not in sql for sql, _ in fake_evil.calls)
      and any(evil in params for _, params in fake_evil.calls))

# 5. Private columns and SELECT * never appear in any query, across all three functions.
fake_all = FakeDB(sold_rows=rows, trend_rows=[{"month": "2026-04", "sales": 1, "avg_price": 1, "avg_ppsf": 1, "avg_dom": 1}],
                  active=5, sold_count=5)
for fn in (stats.city_summary, stats.price_trend, stats.inventory):
    run(fn, fake_all, "Irvine")
banned = ["PrivateRemarks", "ShowingInstructions", "AccessCode", "lockbox", "OwnerName", "OwnerPhone", "ExpirationDate"]
all_sql = " ".join(sql for sql, _ in fake_all.calls).lower()
check("no private columns selected", not any(b.lower() in all_sql for b in banned))
check("no SELECT *", "select *" not in all_sql)

# 6. price_trend: MoM only between consecutive months; floats not Decimals.
trend_rows = [
    {"month": "2026-03", "sales": 10, "avg_price": Decimal("1000000.0"), "avg_ppsf": Decimal("500.4"), "avg_dom": Decimal("20.0")},
    {"month": "2026-04", "sales": 12, "avg_price": Decimal("1100000.0"), "avg_ppsf": Decimal("520.0"), "avg_dom": Decimal("18.5")},
    {"month": "2026-06", "sales": 9, "avg_price": Decimal("990000.0"), "avg_ppsf": None, "avg_dom": None},   # May missing
]
t = run(stats.price_trend, FakeDB(trend_rows=trend_rows), "Irvine")
tr = t["trend"]
check("first month has no MoM", tr[0]["mom_pct"] is None)
check("MoM computed for consecutive months", tr[1]["mom_pct"] == 10.0)
check("MoM is None across a gap month", tr[2]["mom_pct"] is None)
check("trend values are plain numbers", all(not isinstance(v, Decimal) for m in tr for v in m.values()))
check("trend sales are ints", all(isinstance(m["sales"], int) for m in tr))
check("trend handles NULL avg_dom", tr[2]["avg_dom"] is None)
tsql, tparams = FakeDB().calls, None
tf = FakeDB(trend_rows=trend_rows)
run(stats.price_trend, tf, "Irvine", months=24)
tsql, tparams = tf.calls[-1]
check("trend query is parameterized and filtered",
      "City = %s" in tsql and "PropertySubType = %s" in tsql and "Irvine" in tparams)
check("trend window is 24 months back from newest sale", tparams[2] == "2024-09-17" and tparams[3] == "2026-09-17")

# 6b. $/sqft trend, YoY, ZIP.
check("trend carries avg $/sqft as a number", tr[0]["avg_price_per_sqft"] == 500 and tr[2]["avg_price_per_sqft"] is None)
check("YoY None without a year of history", all(m["yoy_pct"] is None for m in tr) and t["yoy_available"] is False)
yoy_rows = [
    {"month": "2025-04", "sales": 5, "avg_price": 1000000.0, "avg_ppsf": 1, "avg_dom": 1},
    {"month": "2026-04", "sales": 5, "avg_price": 1050000.0, "avg_ppsf": 1, "avg_dom": 1},
]
ty = run(stats.price_trend, FakeDB(trend_rows=yoy_rows), "Irvine")
check("YoY computed vs same month last year", ty["trend"][1]["yoy_pct"] == 5.0 and ty["yoy_available"] is True)
zf = FakeDB(sold_rows=rows)
zs = run(stats.city_summary, zf, "Pasadena", zip_code="91107-1234")
zsql, zparams = zf.calls[-1]
check("ZIP is a parameter, trimmed to 5 digits", "LEFT(PostalCode, 5) = %s" in zsql and zparams[-1] == "91107" and "91107" not in zsql)
check("ZIP echoed in result", zs["zip_code"] == "91107")
zt = FakeDB(trend_rows=trend_rows)
run(stats.price_trend, zt, "Pasadena", zip_code="91107")
check("trend honors ZIP", "LEFT(PostalCode, 5) = %s" in zt.calls[-1][0])
zi = FakeDB(active=3, sold_count=6)
run(stats.inventory, zi, "Pasadena", zip_code="91107")
check("inventory ZIP applies to both tables", all("LEFT(" in c[0] for c in zi.calls if "FROM rets_property" in c[0] or "COUNT(*) AS n FROM california_sold" in c[0]))
bad = run(stats.city_summary, FakeDB(sold_rows=rows), "Pasadena", zip_code="9110x")
check("malformed ZIP -> no data, not an unfiltered city result", bad["has_data"] is False)

# 7. inventory.
inv = run(stats.inventory, FakeDB(active=60, sold_count=120), "Irvine", months=12)
check("inventory counts", inv["active_count"] == 60 and inv["sold_count"] == 120)
check("sold pace uses months of data actually covered (6.0), not 12",
      inv["months_covered"] == 6.0 and inv["sold_per_month"] == 20.0)
check("months of supply = active / monthly pace", inv["months_of_supply"] == 3.0)
inv_f = FakeDB(active=60, sold_count=120)
run(stats.inventory, inv_f, "Irvine")
active_sql, active_params = [c for c in inv_f.calls if "rets_property" in c[0]][0]
check("inventory filters StandardStatus on rets_property", "StandardStatus = %s" in active_sql and "Active" in active_params)
check("inventory filters subtype on rets_property", "L_Type_ = %s" in active_sql and "SingleFamilyResidence" in active_params)
inv0 = run(stats.inventory, FakeDB(active=10, sold_count=0), "Irvine")
check("active listings but zero sales: supply is None, no ZeroDivisionError",
      inv0["has_data"] and inv0["months_of_supply"] is None)

# 8. Empty cases return a clean "no data", never a crash.
for name, fn in (("summary", stats.city_summary), ("trend", stats.price_trend), ("inventory", stats.inventory)):
    out = run(fn, FakeDB(), "Atlantis")
    check(f"{name}: no sales -> has_data False with message", out["has_data"] is False and "Atlantis" in out["message"])
    out = run(fn, FakeDB(window=(None, None)), "Irvine")
    check(f"{name}: empty table -> no data", out["has_data"] is False)
    out = run(fn, FakeDB(), "   ")
    check(f"{name}: blank city -> no data", out["has_data"] is False)
    out = run(fn, FakeDB(), None)
    check(f"{name}: None city -> no data", out["has_data"] is False)

# 9. Live, read-only checks against MySQL (skipped when the DB can't be reached).
try:
    from db.connection import query as real_query
    real_query("SELECT 1")
    live = True
except Exception as e:  # no server, bad creds, etc.
    live = False
    print(f"SKIP live checks (database unreachable: {type(e).__name__})")

if live:
    s = stats.city_summary("San Diego")
    check("live: San Diego SFR summary has data", s["has_data"] and s["sold_count"] > 100)
    check("live: median below mean for skewed prices", s["median_close_price"] < s["avg_close_price"])
    check("live: summary is JSON-serializable", bool(json.dumps(s)))
    direct = real_query(
        "SELECT COUNT(*) AS n FROM california_sold WHERE City = %s AND PropertyType = 'Residential' "
        "AND PropertySubType = %s AND ClosePrice > 0 AND CloseDate > %s AND CloseDate <= %s",
        ["San Diego", "SingleFamilyResidence", s["window_start"], s["window_end"]],
    )[0]["n"]
    check("live: sold_count matches an independent COUNT(*)", s["sold_count"] == direct)
    check("live: window ignores typo dates", s["window_end"] <= str(__import__("datetime").date.today()))
    t = stats.price_trend("San Diego")
    check("live: trend months sum to sold total", sum(m["sales"] for m in t["trend"]) == s["sold_count"] or s["months"] != t["months"])
    check("live: some MoM values computed", any(m["mom_pct"] is not None for m in t["trend"]))
    pz = stats.city_summary("Pasadena", zip_code="91107")
    check("live: Pasadena 91107 is a subset of Pasadena", pz["has_data"] and pz["sold_count"] < stats.city_summary("Pasadena")["sold_count"])
    check("live: trend has $/sqft", all(m["avg_price_per_sqft"] for m in t["trend"]))
    check("live: inventory has data", stats.inventory("San Diego")["has_data"])
    check("live: unknown city -> no data", stats.city_summary("Nowhereville")["has_data"] is False)
    condo, sfr = stats.city_summary("Irvine", "Condominium"), stats.city_summary("Irvine")
    check("live: condo and SFR are separate populations", condo["median_close_price"] != sfr["median_close_price"])

print()
print(f"{len(failures)} failure(s)" if failures else "All checks passed")
sys.exit(1 if failures else 0)
