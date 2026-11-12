"""
Week 7: hybrid recommendation engine. Given an active listing the user likes, return the top 5
similar active listings from rets_property, each with a comp-validated price check from california_sold.

    recommend(listing_id, top_k=5, mode="hybrid")  -> dict with the target, recommendations and notes
    validate_with_comps(city, subtype, sqft, price, window) -> sold-comp price check (never divides by zero)
    duplicate_report()                              -> how many listings have near-identical remarks

Score (0-100, same point maxes as the handbook):
    price 20 + beds 15 + city 15 + sqft 10 + semantic 40
Changes from the handbook sample, each for a measured or stated reason:
    price / sqft  relative gap instead of absolute $ / sqft tiers: full points at 0%, zero at 40% apart
    beds          graded: 15 / 10 / 5 / 0 for 0 / 1 / 2 / 3+ beds apart (was exact-or-nothing)
    type          hard filter on L_Type_ (a condo is never offered a house)
    semantic      cosine rescaled with fixed bounds 0.30 -> 0, 0.90 -> 40. Real MiniLM cosines between
                  listings have p10 ~0.35 and p90 ~0.59, so raw cos * 40 would only span ~8 points
                  across 80% of candidates.
All candidates are scored in one vectorised numpy pass (no per-row Python loop, no prefilter that
could miss a good match).

Usage (no Gemini or other API calls; the index is already on disk):
    python recommend/recommend.py 368235                 hybrid top 5 with comp checks
    python recommend/recommend.py 368235 --mode both     pure semantic vs hybrid
    python recommend/recommend.py 368235 --json
    python recommend/recommend.py --dup-report           near-duplicate remarks, same property or not
"""

import argparse
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `from db.connection import query` works when run as a script

from db.connection import query  # noqa: E402
from embeddings.store import load_index  # noqa: E402
from market.stats import _data_window  # noqa: E402
from search.search_listings import PUBLIC_COLUMNS, format_address, money  # noqa: E402

# Point maxes: identical to the handbook so scores stay recognisable.
W_PRICE, W_BEDS, W_CITY, W_SQFT, W_SEM = 20, 15, 15, 10, 40
STRUCT_MAX = W_PRICE + W_BEDS + W_CITY + W_SQFT  # 60

REL_ZERO = 0.40  # price / sqft gap (as a fraction of the target's) that scores 0
BEDS_ZERO = 3  # bed-count gap that scores 0
COS_FLOOR, COS_CEIL = 0.30, 0.90  # cosine mapped to 0 and to full semantic points
NEUTRAL_COS = 0.47  # median cosine between listings; used for candidates with no vector
DUP_COS = 0.99  # cosine at or above which we check whether two listings are the same property
MIN_PRICE = 10_000  # list prices below this are placeholders (18 of 55k listings)

MIN_COMPS = 5  # below this we say "insufficient comps" and print no delta
SQFT_BAND = 0.20  # comps must have LivingArea within +/-20% of the listing
COMP_MONTHS = 6
IN_LINE_PCT, WELL_PCT = 5, 15  # verdict bands on delta_pct


# ---------- listings and index ----------

def _norm_addr(value):
    """Lower-case address with punctuation and unit words (unit, apt, #...) removed."""
    s = re.sub(r"\b(unit|apt|apartment|suite|ste|no)\b\.?|#", " ", str(value or "").lower())
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s).split())


def _norm_parcel(value):
    return re.sub(r"[^0-9a-z]", "", str(value or "").lower())


@lru_cache(maxsize=1)
def _load():
    """(DataFrame of all active listings with a '_vec' row into the index or -1, vectors)."""
    vectors, ids, _ = load_index()
    rows = query(f"SELECT {', '.join(PUBLIC_COLUMNS)}, ParcelNumber FROM rets_property")
    df = pd.DataFrame(rows)
    for col in ("L_SystemPrice", "L_Keyword2", "LM_Dec_3", "LM_Int2_3"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["_addr"] = df["L_Address"].map(_norm_addr)
    df["_city"] = df["L_City"].fillna("").str.strip().str.lower()
    df["_parcel"] = df["ParcelNumber"].map(_norm_parcel)
    row_of = {listing_id: n for n, listing_id in enumerate(ids)}
    df["_vec"] = df["id"].map(row_of).fillna(-1).astype(int)
    return df, vectors


def _same_property(addr_a, city_a, parcel_a, addr_b, city_b, parcel_b):
    """
    Elementwise (numpy arrays ok): True where two listings are the same property.
    Same normalised address and city, unless both have a parcel number and the parcels differ
    (a different parcel with the same address string is a different unit). Different unit
    numbers give different addresses, so same-building neighbours are NOT matched.
    """
    same_addr = (addr_a == addr_b) & (addr_a != "") & (city_a == city_b)
    parcel_conflict = (parcel_a != "") & (parcel_b != "") & (parcel_a != parcel_b)
    return same_addr & ~parcel_conflict


def duplicate_report():
    """Near-duplicate remarks (cosine >= DUP_COS) across the whole index, split by same property or not."""
    df, vectors = _load()
    ids = load_index()[1]
    meta = df.set_index("id").reindex(ids)
    addr, city, parcel = meta["_addr"].fillna("").to_numpy(), meta["_city"].fillna("").to_numpy(), meta["_parcel"].fillna("").to_numpy()
    same_pairs = other_pairs = 0
    same_listings = set()
    for start in range(0, len(vectors), 4000):
        block = vectors[start:start + 4000] @ vectors.T
        rows, cols = np.where(block >= DUP_COS)
        rows = rows + start
        keep = rows < cols  # each pair once, no self-pairs
        rows, cols = rows[keep], cols[keep]
        same = _same_property(addr[rows], city[rows], parcel[rows], addr[cols], city[cols], parcel[cols])
        same_pairs += int(same.sum())
        other_pairs += int((~same).sum())
        same_listings.update(rows[same].tolist() + cols[same].tolist())
    return {
        "pairs_at_or_above_cosine": same_pairs + other_pairs,
        "cosine_threshold": DUP_COS,
        "same_property_pairs_excluded": same_pairs,
        "different_property_pairs_kept": other_pairs,
        "listings_in_a_same_property_pair": len(same_listings),
        "indexed_listings": len(ids),
    }


# ---------- scoring ----------

def _num(value):
    """Plain float, or None for NaN/None."""
    return None if value is None or pd.isna(value) else float(value)


def _linear(gap, zero_at):
    """1.0 at no gap, falling linearly to 0.0 at `zero_at`; NaN gap -> 0."""
    return np.nan_to_num(np.clip(1 - gap / zero_at, 0, 1), nan=0.0)


def _score_all(df, target, tvec, vectors):
    """Score every row of `df` against the target in one pass. Returns a dict of per-row arrays."""
    t_price, t_beds, t_sqft = target["L_SystemPrice"], target["L_Keyword2"], target["LM_Int2_3"]
    price, beds, sqft = (df[c].to_numpy(float) for c in ("L_SystemPrice", "L_Keyword2", "LM_Int2_3"))
    n = len(df)

    price_ok = pd.notna(t_price) and t_price >= MIN_PRICE
    price_pts = W_PRICE * _linear(np.abs(price - t_price) / t_price, REL_ZERO) if price_ok else np.zeros(n)
    beds_pts = W_BEDS * _linear(np.abs(beds - t_beds), BEDS_ZERO) if pd.notna(t_beds) else np.zeros(n)
    t_city = target["_city"]
    city_pts = W_CITY * ((df["_city"].to_numpy() == t_city) & (t_city != ""))
    sqft_ok = pd.notna(t_sqft) and t_sqft > 0
    sqft_gap = np.where(sqft > 0, np.abs(sqft - t_sqft) / t_sqft, np.nan) if sqft_ok else np.full(n, np.nan)
    sqft_pts = W_SQFT * _linear(sqft_gap, REL_ZERO)

    vrow = df["_vec"].to_numpy()
    has_vec = vrow >= 0
    cos = np.full(n, np.nan)
    if tvec is not None:
        cos[has_vec] = vectors[vrow[has_vec]] @ tvec
    cos_used = np.where(has_vec, cos, NEUTRAL_COS) if tvec is not None else np.full(n, np.nan)
    sem_pts = W_SEM * np.clip((cos_used - COS_FLOOR) / (COS_CEIL - COS_FLOOR), 0, 1) if tvec is not None else np.zeros(n)

    structured = price_pts + beds_pts + city_pts + sqft_pts
    # No target vector: structured score only, stretched to 0-100 so it stays comparable in size.
    total = structured + sem_pts if tvec is not None else structured / STRUCT_MAX * 100
    return {
        "total": total, "cos": cos, "has_vec": has_vec, "price": price_pts, "beds": beds_pts,
        "city": city_pts, "sqft": sqft_pts, "semantic": sem_pts,
    }


# ---------- comp validation ----------

def validate_with_comps(city, subtype, sqft, price, window):
    """
    Compare a list price with sold $/sqft for the same city and subtype over `window`
    (the tuple from market.stats._data_window). Uses the median $/sqft of sales with LivingArea within
    +/-20% of `sqft`. Fewer than MIN_COMPS comps -> status "insufficient" and no delta.
    """
    out = {
        "status": "insufficient", "reason": None, "list_price": _num(price), "comp_count": 0,
        "comp_price": None, "delta_pct": None, "verdict": None, "window_start": None, "window_end": None,
    }
    if window is None:
        out["reason"] = "no sold data"
        return out
    start, end, covered, data_start = window
    out["window_start"], out["window_end"], out["months_covered"] = data_start, end, covered
    if not city:
        out["reason"] = "listing has no city"
        return out
    if sqft is None or pd.isna(sqft) or sqft <= 0:
        out["reason"] = "listing has no square footage"
        return out

    where = (
        "City = %s AND PropertyType = 'Residential' AND ClosePrice > 0 AND LivingArea BETWEEN %s AND %s "
        "AND CloseDate > %s AND CloseDate <= %s"
    )
    params = [city, sqft * (1 - SQFT_BAND), sqft * (1 + SQFT_BAND), start, end]
    if subtype:  # None only when the listing's own type is unknown
        where += " AND PropertySubType = %s"
        params.append(subtype)
    rows = query(f"SELECT ClosePrice, LivingArea FROM california_sold WHERE {where}", params)
    comps = pd.DataFrame(rows).astype(float) if rows else pd.DataFrame(columns=["ClosePrice", "LivingArea"])
    out["comp_count"] = int(len(comps))
    if len(comps) < MIN_COMPS:
        out["reason"] = f"only {len(comps)} comps (need {MIN_COMPS})"
        return out

    comp_price = float((comps["ClosePrice"] / comps["LivingArea"]).median() * sqft)  # LivingArea > 0 by the query
    out["comp_price"] = round(comp_price)
    if pd.isna(price) or price < MIN_PRICE:
        out["status"] = "no_price"
        out["reason"] = "list price looks like a placeholder, no delta"
        return out
    delta = (price - comp_price) / comp_price * 100  # comp_price > 0 since every ClosePrice > 0
    out.update(status="ok", delta_pct=round(delta, 1), verdict=_verdict(delta))
    return out


def _verdict(delta):
    if abs(delta) <= IN_LINE_PCT:
        return "in line with comps"
    side = "above" if delta > 0 else "below"
    return f"well {side} comps" if abs(delta) > WELL_PCT else f"{side} comps"


# ---------- main entry ----------

def _card(row):
    """Plain-dict listing fields (no NaN / numpy types)."""
    return {
        "id": int(row["id"]), "address": format_address(row), "city": row["L_City"], "type": row["L_Type_"],
        "price": _num(row["L_SystemPrice"]), "beds": _num(row["L_Keyword2"]), "baths": _num(row["LM_Dec_3"]),
        "sqft": _num(row["LM_Int2_3"]), "hoa": _num(row["AssociationFee"]), "days_on_market": _num(row["DaysOnMarket"]),
    }


def recommend(listing_id, top_k=5, mode="hybrid"):
    """
    Top `top_k` listings similar to `listing_id`. mode="hybrid" is the full score; mode="semantic" ranks by
    raw remark cosine alone (no type filter, for comparison). Raises ValueError for an unknown id.
    """
    if mode not in ("hybrid", "semantic"):
        raise ValueError("mode must be 'hybrid' or 'semantic'")
    top_k = max(1, min(int(top_k), 50))
    df, vectors = _load()
    match = df.index[df["id"] == listing_id]
    if len(match) == 0:
        raise ValueError(f"Listing {listing_id} is not in rets_property.")
    target = df.loc[match[0]]
    tvec = vectors[target["_vec"]] if target["_vec"] >= 0 else None
    if mode == "semantic" and tvec is None:
        raise ValueError(f"Listing {listing_id} has no remarks vector, so pure semantic ranking is unavailable.")

    flags = []
    t_type = target["L_Type_"] if isinstance(target["L_Type_"], str) and target["L_Type_"] else None
    if mode == "hybrid":
        if t_type is None:
            flags.append("type unknown: not filtered by type")
        if pd.isna(target["L_SystemPrice"]) or target["L_SystemPrice"] < MIN_PRICE:
            flags.append(f"list price under ${MIN_PRICE:,} looks like a placeholder: price not scored")
        if pd.isna(target["L_Keyword2"]):
            flags.append("beds missing: beds not scored")
        if pd.isna(target["LM_Int2_3"]) or target["LM_Int2_3"] <= 0:
            flags.append("sqft missing: sqft not scored")
        if tvec is None:
            flags.append("no remarks vector: structured similarity only, rescaled to 100")

    s = _score_all(df, target, tvec, vectors)
    ranking = s["cos"] if mode == "semantic" else s["total"]

    eligible = (df["id"] != listing_id).to_numpy().copy()  # pandas hands back a read-only view
    if mode == "semantic":
        eligible &= s["has_vec"]
    elif t_type is not None:
        eligible &= (df["L_Type_"] == t_type).to_numpy()

    # Same-property check only where remarks are near-identical (a relist or duplicate feed row).
    near = eligible & (np.nan_to_num(s["cos"], nan=0.0) >= DUP_COS)
    same = near & _same_property(
        df["_addr"].to_numpy(), df["_city"].to_numpy(), df["_parcel"].to_numpy(),
        target["_addr"], target["_city"], target["_parcel"],
    )
    eligible &= ~same

    idx = np.flatnonzero(eligible)
    cos_for_sort = np.nan_to_num(s["cos"][idx], nan=-1.0)
    order = idx[np.lexsort((df["id"].to_numpy()[idx], -cos_for_sort, -ranking[idx]))][:top_k]

    window = _data_window(COMP_MONTHS)
    t_card = _card(target)
    t_card["comps"] = validate_with_comps(target["L_City"], t_type, target["LM_Int2_3"], target["L_SystemPrice"], window)
    recs = []
    for i in order:
        row = df.loc[df.index[i]]
        rec = _card(row)
        rec.update(
            score=round(float(ranking[i]), 2),
            cosine=None if np.isnan(s["cos"][i]) else round(float(s["cos"][i]), 3),
            components={k: round(float(s[k][i]), 1) for k in ("price", "beds", "city", "sqft", "semantic")},
        )
        rec_flags = []
        if mode == "hybrid":
            if not s["has_vec"][i] and tvec is not None:
                rec_flags.append("no remarks vector: neutral semantic score")
            if pd.isna(row["LM_Int2_3"]) or row["LM_Int2_3"] <= 0:
                rec_flags.append("sqft missing")
            if pd.isna(row["L_Keyword2"]):
                rec_flags.append("beds missing")
        rec["flags"] = rec_flags
        r_type = row["L_Type_"] if isinstance(row["L_Type_"], str) and row["L_Type_"] else None
        rec["comps"] = validate_with_comps(row["L_City"], r_type, row["LM_Int2_3"], row["L_SystemPrice"], window)
        recs.append(rec)

    checked = [t_card] + recs
    return {
        "mode": mode, "target": t_card, "flags": flags, "recommendations": recs,
        "candidates_scored": int(eligible.sum()),
        "excluded_same_property": int(same.sum()),
        "comp_window": None if window is None else {"start": window[3], "end": window[1], "months": window[2]},
        "comps_insufficient": sum(c["comps"]["status"] == "insufficient" for c in checked),
        "comps_total": len(checked),
    }


# ---------- text output ----------

def _comp_line(c):
    if c["status"] == "insufficient":
        return f"insufficient comps ({c['reason']})"
    if c["status"] == "ok":
        return (f"{c['verdict']}: comps imply {money(c['comp_price'])} vs list {money(c['list_price'])} "
                f"({c['delta_pct']:+.1f}%, {c['comp_count']} comps)")
    return f"{c['comp_count']} comps imply {money(c['comp_price'])}; {c['reason']}"


def _facts(c):
    sqft = f"{c['sqft']:,.0f}" if c["sqft"] else "n/a"
    beds = f"{c['beds']:.0f}" if c["beds"] is not None else "n/a"
    baths = f"{c['baths']:g}" if c["baths"] is not None else "n/a"
    return f"{money(c['price'])} | {beds} bd / {baths} ba | {sqft} sqft | {c['type'] or 'type n/a'}"


def format_result(res):
    t = res["target"]
    lines = [f"[{res['mode']}] Target #{t['id']}: {t['address']}", f"  {_facts(t)}", f"  Comps: {_comp_line(t['comps'])}"]
    lines += [f"  NOTE: {f}" for f in res["flags"]]
    w = res["comp_window"]
    lines.append(
        f"  Scored {res['candidates_scored']:,} candidates; excluded {res['excluded_same_property']} same-property duplicate(s). "
        + (f"Comp window {w['start']} to {w['end']} ({w['months']} months of data)." if w else "No sold data.")
    )
    for n, r in enumerate(res["recommendations"], 1):
        c = r["components"]
        cos = f"cos {r['cosine']:.3f}" if r["cosine"] is not None else "cos n/a"
        lines += [
            f"\n{n}. {r['address']}  (score {r['score']}, {cos})",
            f"   #{r['id']} | {_facts(r)}",
        ]
        if res["mode"] == "hybrid":
            lines.append(f"   price {c['price']} + beds {c['beds']} + city {c['city']} + sqft {c['sqft']} + semantic {c['semantic']}")
        lines.append(f"   Comps: {_comp_line(r['comps'])}")
        lines += [f"   NOTE: {f}" for f in r["flags"]]
    lines.append(f"\nComps insufficient for {res['comps_insufficient']} of {res['comps_total']} listings checked (target + recommendations).")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Recommend similar active listings.")
    ap.add_argument("listing_id", nargs="?", type=int, help="id of a listing in rets_property")
    ap.add_argument("--mode", choices=["hybrid", "semantic", "both"], default="hybrid")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--json", action="store_true", help="raw JSON (use this when answering for a user)")
    ap.add_argument("--dup-report", action="store_true", help="count near-duplicate remarks across the index")
    args = ap.parse_args()

    if args.dup_report:
        print(json.dumps(duplicate_report(), indent=2))
        return
    if args.listing_id is None:
        ap.error("give a listing id")
    modes = ["semantic", "hybrid"] if args.mode == "both" else [args.mode]
    try:
        results = [recommend(args.listing_id, args.top_k, m) for m in modes]
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    if args.json:
        print(json.dumps(results if len(results) > 1 else results[0], indent=2, default=str))
    else:
        print("\n\n".join(format_result(r) for r in results))
        if len(results) == 2:
            a, b = ({r["id"] for r in res["recommendations"]} for res in results)
            print(f"\nOverlap between the two lists: {len(a & b)} of {len(b)} listings.")


if __name__ == "__main__":
    main()
