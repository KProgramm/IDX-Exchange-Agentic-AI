"""
Build a local embedding index of active listings' remarks (all-MiniLM-L6-v2, no paid APIs).

Each listing becomes one line: a short structured header, then the cleaned remarks. The model
truncates at 256 tokens (~1000 chars), so the header stays compact and the remarks come last.

Output, in data/embeddings/:
    vectors.npy  float32 matrix, one L2-normalised row per listing
    ids.json     parallel list of listing ids (rets_property.id)
    meta.json    model, dimension, row count, char-length filter, build timestamp

Resumable: a checkpoint is written every CHECKPOINT_EVERY rows, and a restart skips ids already
in ids.json.

Usage:
    python embeddings/build_index.py              (first 20000 matching listings)
    python embeddings/build_index.py --limit 500
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `from db.connection import query` works when run as a script

from db.connection import query  # noqa: E402

MODEL_NAME = "all-MiniLM-L6-v2"
MIN_REMARKS_CHARS = 80
CHECKPOINT_EVERY = 2000
BATCH_SIZE = 64
OUT_DIR = ROOT / "data" / "embeddings"


def clean_remarks(text):
    """Collapse every whitespace run (including literal \\r\\r\\n) to one space."""
    return re.sub(r"\s+", " ", text).strip()


def build_header(row):
    """One compact line like 'Residential | Irvine | 3 bd | 2 ba | 1,478 sqft | built 1981 | $850,000'."""
    parts = []
    if row["L_Type_"]:
        parts.append(str(row["L_Type_"]).strip())
    if row["L_City"]:
        parts.append(str(row["L_City"]).strip())
    if row["L_Keyword2"] is not None:
        parts.append(f"{row['L_Keyword2']} bd")
    if row["LM_Dec_3"] is not None:
        parts.append(f"{float(row['LM_Dec_3']):g} ba")
    if row["LM_Int2_3"]:  # NULL or 0 means unknown
        parts.append(f"{row['LM_Int2_3']:,} sqft")
    if row["YearBuilt"]:
        parts.append(f"built {row['YearBuilt']}")
    if row["L_SystemPrice"]:
        parts.append(f"${row['L_SystemPrice']:,}")
    return " | ".join(parts)


def embedding_text(row):
    header = build_header(row)
    remarks = clean_remarks(row["L_Remarks"])
    return f"{header}. {remarks}" if header else remarks


def fetch_listings(limit):
    # Every row is Active, so there is deliberately no status filter. ORDER BY id keeps the
    # first `limit` rows the same on every run, which is what makes resuming safe.
    return query(
        "SELECT id, L_Type_, L_City, L_Keyword2, LM_Dec_3, LM_Int2_3, YearBuilt, "
        "L_SystemPrice, L_Remarks FROM rets_property "
        "WHERE CHAR_LENGTH(L_Remarks) >= %s ORDER BY id LIMIT %s",
        [MIN_REMARKS_CHARS, int(limit)],
    )


def write_atomic(path, write_fn):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write_fn(f)
    os.replace(tmp, path)


def save(vectors, ids, dim, complete):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_atomic(OUT_DIR / "vectors.npy", lambda f: np.save(f, vectors))
    write_atomic(OUT_DIR / "ids.json", lambda f: f.write(json.dumps(ids).encode()))
    meta = {
        "model": MODEL_NAME,
        "dimension": dim,
        "row_count": len(ids),
        "min_remarks_chars": MIN_REMARKS_CHARS,
        "complete": complete,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_atomic(OUT_DIR / "meta.json", lambda f: f.write(json.dumps(meta, indent=2).encode()))
    return meta


def load_existing():
    """Return (vectors, ids) from a previous run, or (None, [])."""
    try:
        ids = json.loads((OUT_DIR / "ids.json").read_text())
        vectors = np.load(OUT_DIR / "vectors.npy")
        meta = json.loads((OUT_DIR / "meta.json").read_text())
    except FileNotFoundError:
        return None, []
    if meta["model"] != MODEL_NAME or meta["min_remarks_chars"] != MIN_REMARKS_CHARS:
        sys.exit(f"Existing index in {OUT_DIR} was built with different settings; delete it to rebuild.")
    if len(ids) != len(vectors):
        sys.exit(f"Existing index is inconsistent ({len(ids)} ids, {len(vectors)} vectors); delete it to rebuild.")
    return vectors, ids


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def main():
    ap = argparse.ArgumentParser(description="Build the listing-remarks embedding index.")
    ap.add_argument("--limit", type=int, default=20000, help="max listings to index (default 20000)")
    args = ap.parse_args()

    rows = fetch_listings(args.limit)
    print(f"Fetched {len(rows)} listings with remarks >= {MIN_REMARKS_CHARS} chars")

    vectors, ids = load_existing()
    done = set(str(i) for i in ids)
    todo = [r for r in rows if str(r["id"]) not in done]
    if ids:
        print(f"Resuming: {len(ids)} already embedded, {len(todo)} to go")
    if not todo:
        print("Nothing to do.")
        return

    from sentence_transformers import SentenceTransformer  # slow import, so only when needed

    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    ids = list(ids)
    chunks = [] if vectors is None else [vectors]
    since_checkpoint = 0
    start = time.time()
    total = len(todo)

    for i in range(0, total, BATCH_SIZE):
        batch = todo[i:i + BATCH_SIZE]
        vecs = model.encode([embedding_text(r) for r in batch], normalize_embeddings=True)
        chunks.append(vecs.astype(np.float32))
        ids.extend(r["id"] for r in batch)
        since_checkpoint += len(batch)

        n = min(i + BATCH_SIZE, total)
        elapsed = time.time() - start
        rate = n / elapsed
        print(f"\r{n}/{total}  {rate:.1f} rows/s  ETA {fmt_time((total - n) / rate)}", end="", flush=True)

        if since_checkpoint >= CHECKPOINT_EVERY and n < total:
            save(np.vstack(chunks), ids, dim, complete=False)
            chunks = [np.vstack(chunks)]  # keep one block so later vstacks stay cheap
            since_checkpoint = 0

    meta = save(np.vstack(chunks), ids, dim, complete=True)
    print(f"\nDone in {fmt_time(time.time() - start)}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
