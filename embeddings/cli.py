"""
Spot-check CLI for semantic search over listing remarks. Uses no Gemini or other paid API calls.

Usage:
    python embeddings/cli.py "charming craftsman with mountain views"
    python embeddings/cli.py "quiet cul-de-sac, great for kids" --top-k 10
    python embeddings/cli.py "modern open concept" --json
"""

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from embeddings.build_index import clean_remarks  # noqa: E402
from embeddings.store import search  # noqa: E402
from search.search_listings import format_address  # noqa: E402

SNIPPET_CHARS = 150


def money(value):
    return f"${value:,.0f}" if value is not None else "n/a"


def snippet(row):
    return clean_remarks(row["L_Remarks"] or "")[:SNIPPET_CHARS]


def format_result(rank, row):
    sqft = f"{row['LM_Int2_3']:,}" if row["LM_Int2_3"] is not None else "n/a"
    return (
        f"{rank}. [{row['similarity']:.3f}] {format_address(row)}\n"
        f"   {money(row['L_SystemPrice'])} | {row['L_Keyword2']} bd / {row['LM_Dec_3']} ba | {sqft} sqft | {row['L_Type_']}\n"
        f"   {snippet(row)}"
    )


def to_json_row(row):
    out = {k: v for k, v in row.items() if k != "L_Remarks"}
    out["remarks_snippet"] = snippet(row)
    return out


def json_default(value):
    if isinstance(value, Decimal):
        return float(value)
    return str(value)  # dates etc.


def main():
    ap = argparse.ArgumentParser(description="Semantic search over listing remarks.")
    ap.add_argument("text", help="free-text description of the home")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--json", action="store_true", help="raw results as JSON")
    args = ap.parse_args()

    rows = search(args.text, args.top_k)
    if args.json:
        print(json.dumps([to_json_row(r) for r in rows], indent=2, default=json_default))
    else:
        print(f'Top {len(rows)} for "{args.text}"\n')
        print("\n\n".join(format_result(i, r) for i, r in enumerate(rows, 1)))


if __name__ == "__main__":
    main()
