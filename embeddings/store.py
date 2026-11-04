"""
Semantic search over the listing-remarks index built by embeddings/build_index.py.

    load_index()            -> (vectors, ids, meta), read from disk once and cached
    search(query, top_k=5)  -> top listings by cosine similarity, with details from MySQL

The query is cleaned the same way the remarks were and encoded with the model named in
meta.json. Vectors are L2-normalised, so one matrix-vector dot product gives every cosine score.
"""

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `from db.connection import query` works when run as a script

from db.connection import query  # noqa: E402
from embeddings.build_index import MODEL_NAME, OUT_DIR, clean_remarks  # noqa: E402
from search.search_listings import MAX_LIMIT, PUBLIC_COLUMNS  # noqa: E402

DETAIL_COLUMNS = PUBLIC_COLUMNS + ["L_Remarks"]


@lru_cache(maxsize=1)
def load_index():
    """Return (vectors, ids, meta). Read from disk on the first call, cached after that."""
    try:
        meta = json.loads((OUT_DIR / "meta.json").read_text())
        ids = json.loads((OUT_DIR / "ids.json").read_text())
        vectors = np.load(OUT_DIR / "vectors.npy")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{e.filename} not found. Run embeddings/build_index.py first.") from e

    if meta["model"] != MODEL_NAME:
        raise ValueError(
            f"Index was built with model '{meta['model']}' but this code loads '{MODEL_NAME}'. "
            "Vectors from different models are not comparable; rebuild the index."
        )
    if len(ids) != len(vectors) or vectors.shape[1] != meta["dimension"]:
        raise ValueError(
            f"Index files disagree: {len(ids)} ids, vectors {vectors.shape}, "
            f"meta says {meta['row_count']} x {meta['dimension']}. Rebuild the index."
        )
    return vectors, ids, meta


@lru_cache(maxsize=1)
def _model(name):
    from sentence_transformers import SentenceTransformer  # slow import, so only when searching

    return SentenceTransformer(name)


def search(text, top_k=5):
    """Return up to top_k listing dicts (best first), each with a 'similarity' score."""
    top_k = max(1, min(int(top_k), MAX_LIMIT))
    cleaned = clean_remarks(text or "")
    if not cleaned:
        raise ValueError("Search text is empty.")

    vectors, ids, meta = load_index()
    model = _model(meta["model"])
    if model.get_sentence_embedding_dimension() != vectors.shape[1]:
        raise ValueError("Loaded model's dimension doesn't match the index; rebuild the index.")

    qvec = model.encode(cleaned, normalize_embeddings=True).astype(np.float32)
    scores = vectors @ qvec  # one dot product against the whole matrix

    k = min(top_k, len(scores))
    top = np.argpartition(-scores, k - 1)[:k]  # unordered top k, O(n)
    top = top[np.argsort(-scores[top])]  # then order just those k

    top_ids = [ids[i] for i in top]
    similarity = {ids[i]: float(scores[i]) for i in top}

    rows = query(
        f"SELECT {', '.join(DETAIL_COLUMNS)} FROM rets_property "
        f"WHERE id IN ({', '.join(['%s'] * len(top_ids))})",
        top_ids,
    )
    by_id = {r["id"]: r for r in rows}
    results = []
    for listing_id in top_ids:  # keep similarity order; skip ids no longer in the table
        if listing_id in by_id:
            results.append({**by_id[listing_id], "similarity": similarity[listing_id]})
    return results
