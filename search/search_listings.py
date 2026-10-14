"""
Week 3: search the MySQL database using the filter dict from the Week 2 parser.

Two main functions:
    search_active_listings(filters, page, limit)  -> active listings from rets_property
    get_sold_comps(city, months)                  -> recent sales from california_sold

Usage:
    python search/search_listings.py "3 bed condos in Irvine under $1.5M"   (1 Gemini call)
    python search/search_listings.py --filters '{"city": "Irvine", "pool": false}'  (no Gemini)
    add --page 2 --limit 5 to paginate, or --comps to show recent sold comps for the city
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `from db.connection import query` works when run as a script

from db.connection import query  # noqa: E402

MAX_LIMIT = 50  # hard cap: no call can ever return more than this many rows

# Public columns only. We list them by name instead of using SELECT *, so private
# fields (agent contact info, remarks, photos blob, etc.) can never leak out.
PUBLIC_COLUMNS = [
    "id", "L_Address", "L_City", "L_State", "L_Zip", "L_Type_",
    "L_SystemPrice", "L_Keyword2", "LM_Dec_3", "LM_Int2_3",
    "AssociationFee", "DaysOnMarket", "PoolPrivateYN", "ViewYN", "PhotoCount",
]


def build_where(filters):
    """Turn the filter dict into (list of SQL conditions, list of %s values)."""
    conds = ["L_Status = 'Active'"]  # active listings only
    params = []

    if filters.get("city") is not None:
        conds.append("L_City = %s")
        params.append(filters["city"])
    if filters.get("max_price") is not None:
        conds.append("L_SystemPrice <= %s")
        params.append(filters["max_price"])
    if filters.get("min_beds") is not None:
        conds.append("L_Keyword2 >= %s")
        params.append(filters["min_beds"])
    if filters.get("min_baths") is not None:
        conds.append("LM_Dec_3 >= %s")
        params.append(filters["min_baths"])
    if filters.get("min_sqft") is not None:
        conds.append("LM_Int2_3 >= %s")
        params.append(filters["min_sqft"])
    if filters.get("property_type") is not None:
        conds.append("L_Type_ = %s")
        params.append(filters["property_type"])

    # Pool/view are stored as the string '1' for yes; '' or NULL mean no/unknown.
    # We test "is not None" (not truthiness) so False still filters.
    for key, column in (("pool", "PoolPrivateYN"), ("view", "ViewYN")):
        if filters.get(key) is True:
            conds.append(f"{column} = '1'")
        elif filters.get(key) is False:
            conds.append(f"({column} IS NULL OR {column} <> '1')")

    # Listings with no HOA fee on file (NULL) should still match an HOA limit.
    if filters.get("max_hoa") is not None:
        conds.append("(AssociationFee IS NULL OR AssociationFee <= %s)")
        params.append(filters["max_hoa"])

    return conds, params


def search_active_listings(filters, page=1, limit=10):
    """Return one page of active listings that match the filters."""
    limit = max(1, min(int(limit), MAX_LIMIT))
    page = max(1, int(page))
    conds, params = build_where(filters)

    # ORDER BY newest first, then id: the id tie-break keeps the order stable,
    # so page 1 and page 2 never repeat or skip a listing.
    sql = (
        f"SELECT {', '.join(PUBLIC_COLUMNS)} FROM rets_property "
        f"WHERE {' AND '.join(conds)} "
        "ORDER BY ListingContractDate DESC, id "
        "LIMIT %s OFFSET %s"
    )
    return query(sql, params + [limit, (page - 1) * limit])


def get_sold_comps(city, months=6):
    """Residential sales in `city` during the last `months` months of the data."""
    # The window is measured back from the newest sale in the table, not from today,
    # because the sold data stops at a fixed date. We ignore dates after today:
    # the table has a few typo dates (2028, 2072) that would wreck the window.
    sql = (
        "SELECT UnparsedAddress, City, PropertySubType, BedroomsTotal, "
        "BathroomsTotalInteger, LivingArea, ListPrice, ClosePrice, CloseDate "
        "FROM california_sold "
        "WHERE City = %s AND PropertyType = 'Residential' "
        "AND CloseDate <= (SELECT MAX(CloseDate) FROM california_sold WHERE CloseDate <= CURDATE()) "
        "AND CloseDate > DATE_SUB("
        "(SELECT MAX(CloseDate) FROM california_sold WHERE CloseDate <= CURDATE()), "
        "INTERVAL %s MONTH) "
        "ORDER BY CloseDate DESC "
        "LIMIT %s"
    )
    return query(sql, [city, int(months), MAX_LIMIT])


def money(value):
    return f"${value:,.0f}" if value is not None else "n/a"


def format_property_card(row):
    """One active listing as a short text card."""
    hoa = f"{money(row['AssociationFee'])}/mo" if row["AssociationFee"] is not None else "none listed"
    pool = "yes" if row["PoolPrivateYN"] == "1" else "no"
    sqft = f"{row['LM_Int2_3']:,}" if row["LM_Int2_3"] is not None else "n/a"
    return (
        f"{row['L_Address']}, {row['L_City']}, {row['L_State']} {row['L_Zip']}\n"
        f"  Price: {money(row['L_SystemPrice'])} | {row['L_Keyword2']} bd / {row['LM_Dec_3']} ba | {sqft} sqft\n"
        f"  Type: {row['L_Type_']} | HOA: {hoa} | Days on market: {row['DaysOnMarket']}\n"
        f"  Pool: {pool} | Photos: {row['PhotoCount']}"
    )


def format_comp_card(row):
    """One sold comp as a short text card, with sale-to-list ratio."""
    if row["ListPrice"]:  # skip the division if list price is missing or zero
        ratio = f"{row['ClosePrice'] / row['ListPrice']:.1%}"
    else:
        ratio = "n/a"
    return (
        f"{row['UnparsedAddress']}\n"
        f"  Sold: {money(row['ClosePrice'])} on {row['CloseDate']} | "
        f"List: {money(row['ListPrice'])} | Sale-to-list: {ratio}\n"
        f"  {row['PropertySubType']} | {row['BedroomsTotal']} bd / {row['BathroomsTotalInteger']} ba"
    )


def main():
    ap = argparse.ArgumentParser(description="Search listings in MySQL.")
    ap.add_argument("text", nargs="?", help="free-text search (uses 1 Gemini call)")
    ap.add_argument("--filters", help="filter dict as JSON (uses no Gemini calls)")
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--comps", action="store_true", help="show sold comps for the city")
    ap.add_argument("--months", type=int, default=6, help="comps window in months")
    args = ap.parse_args()

    if args.filters:
        filters = json.loads(args.filters)
    elif args.text:
        # Imported here so the --filters path never needs Gemini installed or configured.
        sys.path.insert(0, str(ROOT / "parser"))
        from parse_query import parse_property_query
        filters = parse_property_query(args.text)
    else:
        ap.error("give a search text or --filters")

    print("Filters:", json.dumps(filters))

    if args.comps:
        if not filters.get("city"):
            ap.error("--comps needs a city")
        rows = get_sold_comps(filters["city"], args.months)
        print(f"\n{len(rows)} sold comps in {filters['city']} (last {args.months} months):\n")
        print("\n\n".join(format_comp_card(r) for r in rows))
    else:
        rows = search_active_listings(filters, args.page, args.limit)
        print(f"\nPage {args.page}: {len(rows)} listings\n")
        print("\n\n".join(format_property_card(r) for r in rows))


if __name__ == "__main__":
    main()
