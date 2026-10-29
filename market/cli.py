"""
Week 5: spot-check CLI for the market stats. Uses no Gemini calls.

Usage:
    python market/cli.py "San Diego"                       summary + trend + inventory
    python market/cli.py Irvine --only summary             one of: summary, trend, inventory
    python market/cli.py Irvine --type Condominium         default type is SingleFamilyResidence
    python market/cli.py Irvine --months 6 --json          raw dicts as JSON
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from market.stats import DEFAULT_SUBTYPE, city_summary, inventory, price_trend  # noqa: E402


def money(value):
    return f"${value:,.0f}" if value is not None else "n/a"


def num(value, suffix=""):
    return f"{value:,}{suffix}" if value is not None else "n/a"


def zip_label(r):
    return f" {r['zip_code']}" if r.get("zip_code") else ""


def format_summary(s):
    return (
        f"{s['city']}{zip_label(s)} | {s['property_subtype']} | {s['window_start']} to {s['window_end']} "
        f"({s['months_covered']} months of data)\n"
        f"  Sold: {s['sold_count']:,}\n"
        f"  Close price: avg {money(s['avg_close_price'])} | median {money(s['median_close_price'])}\n"
        f"  Price/sqft: {money(s['avg_price_per_sqft'])} (from {s['price_per_sqft_sample']:,} sales with sqft)\n"
        f"  Avg days on market: {num(s['avg_dom'])} | List-to-close: {num(s['list_to_close_pct'], '%')}"
    )


def format_trend(t):
    lines = [f"{t['city']}{zip_label(t)} | {t['property_subtype']} | monthly trend (edge months may be partial)"]
    lines.append(f"  {'Month':<8}{'Sales':>6}{'Avg price':>13}{'$/sqft':>8}{'Avg DOM':>9}{'MoM':>8}{'YoY':>8}")
    for m in t["trend"]:
        mom = f"{m['mom_pct']:+.1f}%" if m["mom_pct"] is not None else "-"
        yoy = f"{m['yoy_pct']:+.1f}%" if m["yoy_pct"] is not None else "-"
        lines.append(
            f"  {m['month']:<8}{m['sales']:>6}{money(m['avg_price']):>13}{money(m['avg_price_per_sqft']):>8}"
            f"{num(m['avg_dom']):>9}{mom:>8}{yoy:>8}"
        )
    if not t["yoy_available"]:
        lines.append("  YoY n/a: the sold data covers less than 13 months.")
    return "\n".join(lines)


def format_inventory(i):
    supply = num(i["months_of_supply"], " months") if i["months_of_supply"] is not None else "n/a (no sales)"
    return (
        f"{i['city']}{zip_label(i)} | {i['property_subtype']} | inventory\n"
        f"  Active listings now: {i['active_count']:,}\n"
        f"  Sold in window: {i['sold_count']:,} (~{i['sold_per_month']}/month over {i['months_covered']} months of data)\n"
        f"  Months of supply: {supply}"
    )


def main():
    ap = argparse.ArgumentParser(description="Spot-check market stats for a California city.")
    ap.add_argument("city")
    ap.add_argument("--zip", dest="zip_code", help="narrow the city to one 5-digit ZIP")
    ap.add_argument("--type", default=DEFAULT_SUBTYPE, help="PropertySubType (default %(default)s)")
    ap.add_argument("--months", type=int, default=12, help="summary/inventory window (trend uses --trend-months)")
    ap.add_argument("--trend-months", type=int, default=24)
    ap.add_argument("--only", choices=["summary", "trend", "inventory"])
    ap.add_argument("--json", action="store_true", help="print raw dicts as JSON")
    args = ap.parse_args()

    jobs = {
        "summary": (lambda: city_summary(args.city, args.type, args.months, args.zip_code), format_summary),
        "trend": (lambda: price_trend(args.city, args.type, args.trend_months, args.zip_code), format_trend),
        "inventory": (lambda: inventory(args.city, args.type, args.months, args.zip_code), format_inventory),
    }
    for name, (run, fmt) in jobs.items():
        if args.only and args.only != name:
            continue
        result = run()
        if args.json:
            print(json.dumps({name: result}, indent=2))
        else:
            print(fmt(result) if result["has_data"] else result["message"])
            print()


if __name__ == "__main__":
    main()
