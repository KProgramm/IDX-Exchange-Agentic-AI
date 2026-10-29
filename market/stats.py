"""
Week 5: market statistics over california_sold (sold comps) and rets_property (active inventory).

Three main functions:
    city_summary(city, property_subtype, months)  -> sold count, avg/median price, $/sqft, DOM, list-to-close
    price_trend(city, property_subtype, months)   -> monthly avg price, $/sqft, sales, avg DOM, MoM % and YoY %
    inventory(city, property_subtype, months)     -> active listings vs. sold volume
Each takes an optional zip_code=... to narrow the city to one ZIP.

Every function returns a plain dict of floats/ints/strings/None (no Decimals), so it can go straight
to json.dumps or an f-string. When a city has no sales it returns {"has_data": False, "message": ...}
instead of raising.

Usage:
    python market/cli.py "San Diego"        (no Gemini calls)
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `from db.connection import query` works when run as a script

from db.connection import query  # noqa: E402

# Default subtype. Condos, townhouses and SFRs price very differently, so one subtype per
# call is the rule; there is deliberately no "all residential" mode.
DEFAULT_SUBTYPE = "SingleFamilyResidence"
PROPERTY_TYPE = "Residential"  # california_sold holds only this, but we still filter on it

# California_sold has no StandardStatus column (every row is a closed sale). Active status
# is filtered on rets_property instead.
ACTIVE_STATUS = "Active"

# Columns are listed by name in every query below, never SELECT *, so PrivateRemarks,
# ShowingInstructions, AccessCode, lockbox, OwnerName, OwnerPhone and ExpirationDate
# can never be pulled into a result.


def _num(value, digits=None):
    """Decimal / numpy / NaN -> plain float, int, or None. Rounds floats if `digits` is given."""
    if value is None or pd.isna(value):
        return None
    value = float(value)
    if digits is not None:
        value = round(value, digits)
    return int(value) if digits == 0 else value


def _no_data(city, subtype, months, what="sales"):
    return {
        "has_data": False,
        "city": city,
        "property_subtype": subtype,
        "months": months,
        "message": f"No {subtype} {what} found for '{city}' in the last {months} months.",
    }


def _clean_city(city):
    return (city or "").strip()


def _data_window(months):
    """
    Return (start, end, months_covered) for the sold-data window, or None if there is no data.

    `end` is the newest sale on or before today, not today itself: the sold data stops at a
    fixed date, and the table has a few typo dates (2028, 2072) that must be ignored.
    `start` is exclusive. `months_covered` is how many months of data the window really has,
    which is less than `months` when the table doesn't go back that far.
    """
    rows = query(
        "SELECT MIN(CloseDate) AS first_sale, MAX(CloseDate) AS last_sale "
        "FROM california_sold WHERE CloseDate <= CURDATE() AND ClosePrice > 0"
    )
    if not rows or rows[0]["last_sale"] is None:
        return None
    first, last = pd.Timestamp(rows[0]["first_sale"]), pd.Timestamp(rows[0]["last_sale"])
    start = last - pd.DateOffset(months=int(months))
    covered = (last - max(start, first)).days / 30.44
    return str(start.date()), str(last.date()), round(covered, 1)


# Shared filters. City / subtype / dates always arrive as %s parameters.
SOLD_WHERE = (
    "City = %s AND PropertyType = 'Residential' AND PropertySubType = %s "
    "AND ClosePrice > 0 AND CloseDate > %s AND CloseDate <= %s"
)


def _clean_zip(zip_code):
    """First 5 digits of a ZIP (the table has some ZIP+4 values), or None."""
    z = str(zip_code).strip()[:5] if zip_code else ""
    return z if z.isdigit() and len(z) == 5 else None


def _sold_filter(city, subtype, start, end, zip_code):
    """(WHERE clause, params) for sold sales; adds the ZIP condition when one is given."""
    where, params = SOLD_WHERE, [city, subtype, start, end]
    if zip_code:
        where += " AND LEFT(PostalCode, 5) = %s"
        params.append(zip_code)
    return where, params


def city_summary(city, property_subtype=DEFAULT_SUBTYPE, months=12, zip_code=None):
    """Sold stats for one city and one property subtype over the last `months` months of data."""
    city, zip_raw = _clean_city(city), zip_code
    zip_code = _clean_zip(zip_raw)
    window = _data_window(months) if city and (zip_code or not zip_raw) else None
    if window is None:
        return _no_data(city, property_subtype, months)
    start, end, covered = window

    where, params = _sold_filter(city, property_subtype, start, end, zip_code)
    # We pull the columns and aggregate in pandas rather than in SQL: MySQL has no MEDIAN(),
    # and doing the mean and median over the same fetched rows keeps them consistent.
    rows = query(
        "SELECT ClosePrice, ListPrice, LivingArea, DaysOnMarket "
        f"FROM california_sold WHERE {where}",
        params,
    )
    if not rows:
        return _no_data(city, property_subtype, months)

    df = pd.DataFrame(rows).astype(float)
    # Every divisor is guarded: no price/sqft without sqft, no ratio without a list price.
    sqft = df["LivingArea"].where(df["LivingArea"] > 0)
    list_price = df["ListPrice"].where(df["ListPrice"] > 0)
    dom = df["DaysOnMarket"].where(df["DaysOnMarket"] >= 0)
    ppsf = df["ClosePrice"] / sqft
    ratio = df["ClosePrice"] / list_price

    return {
        "has_data": True,
        "city": city,
        "property_type": PROPERTY_TYPE,
        "property_subtype": property_subtype,
        "zip_code": zip_code,
        "months": months,
        "months_covered": covered,
        "window_start": start,
        "window_end": end,
        "sold_count": int(len(df)),
        "avg_close_price": _num(df["ClosePrice"].mean(), 0),
        "median_close_price": _num(df["ClosePrice"].median(), 0),
        "avg_price_per_sqft": _num(ppsf.mean(), 0),
        "price_per_sqft_sample": int(ppsf.notna().sum()),  # sales that had a usable LivingArea
        "avg_dom": _num(dom.mean(), 1),
        "list_to_close_pct": _num(ratio.mean() * 100, 1),
    }


def _yoy(month, avg_price, by_month):
    """% change in avg price vs. the same month a year earlier, or None if that month isn't in the data."""
    before = by_month.get(str(pd.Period(month) - 12))
    return round((avg_price / before - 1) * 100, 1) if before and avg_price else None


def price_trend(city, property_subtype=DEFAULT_SUBTYPE, months=24, zip_code=None):
    """Monthly avg price, sales count, avg DOM and month-over-month % change."""
    city, zip_raw = _clean_city(city), zip_code
    zip_code = _clean_zip(zip_raw)
    window = _data_window(months) if city and (zip_code or not zip_raw) else None
    if window is None:
        return _no_data(city, property_subtype, months)
    start, end, covered = window

    where, params = _sold_filter(city, property_subtype, start, end, zip_code)
    rows = query(
        "SELECT SUBSTRING(CloseDate, 1, 7) AS month, COUNT(*) AS sales, "
        "AVG(ClosePrice) AS avg_price, "
        "AVG(CASE WHEN LivingArea > 0 THEN ClosePrice / LivingArea END) AS avg_ppsf, "
        "AVG(CASE WHEN DaysOnMarket >= 0 THEN DaysOnMarket END) AS avg_dom "
        f"FROM california_sold WHERE {where} "
        "GROUP BY month ORDER BY month",
        params,
    )
    if not rows:
        return _no_data(city, property_subtype, months)

    by_month = {r["month"]: _num(r["avg_price"], 0) for r in rows}
    out, prev = [], None
    for r in rows:
        avg_price = _num(r["avg_price"], 0)
        mom = None
        # Only compare against the immediately preceding calendar month; a gap month with
        # no sales would otherwise make the "change" span two months.
        if prev and (pd.Period(r["month"]) - pd.Period(prev["month"])).n == 1 and prev["avg_price"]:
            mom = round((avg_price / prev["avg_price"] - 1) * 100, 1)
        entry = {
            "month": r["month"],
            "sales": int(r["sales"]),
            "avg_price": avg_price,
            "avg_price_per_sqft": _num(r["avg_ppsf"], 0),
            "avg_dom": _num(r["avg_dom"], 1),
            "mom_pct": mom,
            "yoy_pct": _yoy(r["month"], avg_price, by_month),
        }
        out.append(entry)
        prev = entry

    return {
        "has_data": True,
        "city": city,
        "property_subtype": property_subtype,
        "zip_code": zip_code,
        "months": months,
        "months_covered": covered,
        "window_start": start,
        "window_end": end,
        "yoy_available": any(m["yoy_pct"] is not None for m in out),  # needs 13+ months of history
        "trend": out,  # first and last month are usually partial (window edges)
    }


def inventory(city, property_subtype=DEFAULT_SUBTYPE, months=12, zip_code=None):
    """Active listings now (rets_property) vs. sold volume over the window (california_sold)."""
    city, zip_raw = _clean_city(city), zip_code
    zip_code = _clean_zip(zip_raw)
    window = _data_window(months) if city and (zip_code or not zip_raw) else None
    if window is None:
        return _no_data(city, property_subtype, months, what="inventory")
    start, end, covered = window

    # rets_property uses its own legacy names: L_City, and L_Type_ for the subtype.
    active_where = "L_City = %s AND StandardStatus = %s AND L_Type_ = %s"
    active_params = [city, ACTIVE_STATUS, property_subtype]
    if zip_code:
        active_where += " AND LEFT(L_Zip, 5) = %s"
        active_params.append(zip_code)
    active = query(f"SELECT COUNT(*) AS n FROM rets_property WHERE {active_where}", active_params)
    where, params = _sold_filter(city, property_subtype, start, end, zip_code)
    sold = query(f"SELECT COUNT(*) AS n FROM california_sold WHERE {where}", params)
    active_count, sold_count = int(active[0]["n"]), int(sold[0]["n"])
    if active_count == 0 and sold_count == 0:
        return _no_data(city, property_subtype, months, what="inventory")

    # Months of supply = how long today's active listings would last at the recent sales pace.
    # The pace uses months_covered (not `months`) so a short data history isn't under-counted.
    per_month = sold_count / covered if covered else 0
    return {
        "has_data": True,
        "city": city,
        "property_subtype": property_subtype,
        "zip_code": zip_code,
        "months": months,
        "months_covered": covered,
        "active_count": active_count,
        "sold_count": sold_count,
        "sold_per_month": _num(per_month, 1),
        "months_of_supply": _num(active_count / per_month, 1) if per_month else None,
    }
