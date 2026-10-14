"""
Week 3 tests for the MySQL query layer. Uses dict filters only: zero Gemini calls.

Usage:
    python tests/test_search.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db.connection import query  # noqa: E402
from search.search_listings import (  # noqa: E402
    MAX_LIMIT, format_comp_card, format_property_card,
    get_sold_comps, search_active_listings,
)

failures = []


def check(name, condition):
    print(("PASS" if condition else "FAIL"), name)
    if not condition:
        failures.append(name)


# The listing rows don't include L_Status, so verify "active only" with a direct count.
total_active = query("SELECT COUNT(*) AS n FROM rets_property WHERE L_Status = 'Active'")[0]["n"]
total_rows = query("SELECT COUNT(*) AS n FROM rets_property")[0]["n"]
check("only active listings exist/are searched", total_active == total_rows)

# 1. Numeric and text filters are respected.
rows = search_active_listings(
    {"city": "Irvine", "max_price": 1500000, "min_beds": 3, "min_baths": 2,
     "min_sqft": 1200, "property_type": "Condominium"}, limit=50)
check("combined filter returns rows", len(rows) > 0)
check("city respected", all(r["L_City"] == "Irvine" for r in rows))
check("max_price respected", all(r["L_SystemPrice"] <= 1500000 for r in rows))
check("min_beds respected", all(r["L_Keyword2"] >= 3 for r in rows))
check("min_baths respected", all(r["LM_Dec_3"] >= 2 for r in rows))
check("min_sqft respected", all(r["LM_Int2_3"] >= 1200 for r in rows))
check("property_type respected", all(r["L_Type_"] == "Condominium" for r in rows))

# 2. pool=True and pool=False both filter (False must not be ignored).
pool_yes = search_active_listings({"city": "Irvine", "pool": True}, limit=50)
pool_no = search_active_listings({"city": "Irvine", "pool": False}, limit=50)
check("pool=True rows all have a pool", len(pool_yes) > 0 and all(r["PoolPrivateYN"] == "1" for r in pool_yes))
check("pool=False rows have no pool", len(pool_no) > 0 and all(r["PoolPrivateYN"] != "1" for r in pool_no))
view_no = search_active_listings({"city": "Irvine", "view": False}, limit=50)
check("view=False rows have no view", len(view_no) > 0 and all(r["ViewYN"] != "1" for r in view_no))

# 3. max_hoa keeps listings with NULL HOA.
hoa = search_active_listings({"max_hoa": 300}, limit=50)
check("max_hoa respected", all(r["AssociationFee"] is None or r["AssociationFee"] <= 300 for r in hoa))
# Single-family homes often have no HOA on file, so NULL rows should show up here.
sfr = search_active_listings({"property_type": "SingleFamilyResidence", "max_hoa": 300}, limit=50)
check("max_hoa includes NULL-HOA listings", any(r["AssociationFee"] is None for r in sfr))

# 4. Pagination: pages don't overlap and page 2 is not empty.
p1 = search_active_listings({"city": "Irvine"}, page=1, limit=10)
p2 = search_active_listings({"city": "Irvine"}, page=2, limit=10)
ids1, ids2 = {r["id"] for r in p1}, {r["id"] for r in p2}
check("page 1 and 2 have 10 rows each", len(p1) == 10 and len(p2) == 10)
check("pages do not overlap", not (ids1 & ids2))

# 5. Hard cap.
many = search_active_listings({}, limit=500)
check(f"limit capped at {MAX_LIMIT}", len(many) == MAX_LIMIT)

# 6. No private columns leak.
banned = {"PrivateRemarks", "ShowingInstructions", "AccessCode", "ExpirationDate", "L_Remarks", "L_Photos"}
check("no private columns returned", not (banned & set(many[0].keys())))

# 7. Sold comps.
comps = get_sold_comps("Irvine", months=6)
check("comps return rows", len(comps) > 0)
check("comps are all Irvine", all(c["City"] == "Irvine" for c in comps))
check("comps are newest first", [c["CloseDate"] for c in comps] == sorted((c["CloseDate"] for c in comps), reverse=True))
check("comps window relative to data", len(get_sold_comps("Irvine", months=1)) <= len(comps))

# 8. Cards render without errors.
check("property card renders", "Irvine" in format_property_card(p1[0]))
check("comp card shows sale-to-list", "Sale-to-list" in format_comp_card(comps[0]))

print(f"\n{'ALL PASSED' if not failures else str(len(failures)) + ' FAILED: ' + ', '.join(failures)}")
sys.exit(1 if failures else 0)
