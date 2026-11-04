"""
Tests for the semantic search index (build_index.py, store.py, cli.py). Zero network calls: the
database and the embedding model are mocked, and the real index on disk is never loaded.

Usage:
    python tests/test_embeddings.py
"""

import contextlib
import io
import json
import sys
import tempfile
import types
from decimal import Decimal
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from embeddings import build_index, cli, store  # noqa: E402
from search.search_listings import format_address, format_property_card  # noqa: E402

failures = []


def check(name, condition):
    print(("PASS" if condition else "FAIL"), name)
    if not condition:
        failures.append(name)


def raises(exc_type, fn, *args, **kwargs):
    """Return the exception message if fn raises exc_type, else None."""
    try:
        fn(*args, **kwargs)
    except exc_type as e:
        return str(e)
    return None


def quiet(fn, *args, **kwargs):
    """Run fn with stdout swallowed (build_index prints progress)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


LONG = "A lovely home with plenty of natural light and a big yard for the whole family to enjoy. "  # > 80 chars


def listing(id_, remarks=LONG, **over):
    row = {"id": id_, "L_Type_": "SingleFamilyResidence", "L_City": "Irvine", "L_Keyword2": 3,
           "LM_Dec_3": Decimal("2.5"), "LM_Int2_3": 1800, "YearBuilt": 1990, "L_SystemPrice": 900000,
           "L_Remarks": remarks}
    row.update(over)
    return row


def full_row(**over):
    """A row shaped like search_active_listings / store.search results."""
    row = {"id": 1, "L_Address": "1 Main St", "L_City": "Irvine", "L_State": "CA", "L_Zip": "92618",
           "L_Type_": "Condominium", "L_SystemPrice": 750000, "L_Keyword2": 2, "LM_Dec_3": Decimal("2.0"),
           "LM_Int2_3": 1200, "AssociationFee": 300, "DaysOnMarket": 12, "PoolPrivateYN": "1",
           "ViewYN": None, "PhotoCount": 20, "L_Remarks": "Bright\r\r\n\r\r\ncorner   unit.", "similarity": 0.5}
    row.update(over)
    return row


class FakeModel:
    """Stands in for SentenceTransformer. Encodes by a fixed text -> vector map and records calls."""

    def __init__(self, dim=3, vectors=None, fail_after=None):
        self.dim, self.vectors, self.fail_after = dim, vectors or {}, fail_after
        self.encoded = []  # every text encoded, in order
        self.normalize_flags = []

    def get_sentence_embedding_dimension(self):
        return self.dim

    def encode(self, texts, normalize_embeddings=False, **_):
        self.normalize_flags.append(normalize_embeddings)
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        if self.fail_after is not None and len(self.encoded) + len(batch) > self.fail_after:
            raise RuntimeError("simulated crash")
        self.encoded.extend(batch)
        out = []
        for t in batch:
            v = np.array(self.vectors.get(t, [1.0 + len(t) % 7, 2.0, 3.0 + len(t) % 5][: self.dim]), dtype=np.float64)
            if normalize_embeddings:
                v = v / np.linalg.norm(v)
            out.append(v)
        out = np.array(out)
        return out[0] if single else out


def fake_sentence_transformers(model):
    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = lambda name: model
    return mod


# ---------------------------------------------------------------- 1. fetch filter (DB mocked)
class FakeDB:
    """Stands in for db.connection.query; applies the CHAR_LENGTH filter the SQL asks for."""

    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def __call__(self, sql, params=None):
        self.calls.append((sql, list(params or [])))
        if "FROM rets_property" in sql and "CHAR_LENGTH(L_Remarks) >= %s" in sql:
            minimum, limit = params
            kept = [r for r in self.rows if r["L_Remarks"] is not None and len(r["L_Remarks"]) >= minimum]
            return sorted(kept, key=lambda r: r["id"])[:limit]
        return []


rows = [listing(1, "x" * 79), listing(2, "x" * 80), listing(3, None), listing(4, LONG), listing(5, "short")]
db = FakeDB(rows)
with mock.patch.object(build_index, "query", db):
    got = build_index.fetch_listings(100)
check("filter constant is 80", build_index.MIN_REMARKS_CHARS == 80)
check("filter passed to SQL as 80", db.calls[0][1][0] == 80)
check("remarks under 80 chars excluded", [r["id"] for r in got] == [2, 4])
check("exactly 80 chars is included", 2 in [r["id"] for r in got])
check("NULL remarks excluded", 3 not in [r["id"] for r in got])
check("SQL has no status filter (all rows are Active)", "Status" not in db.calls[0][0])
with mock.patch.object(build_index, "query", db):
    check("limit is applied", len(build_index.fetch_listings(1)) == 1)

# ---------------------------------------------------------------- 2. whitespace normalisation
check("collapses \\r\\r\\n", build_index.clean_remarks("Nice\r\r\n\r\r\nhome") == "Nice home")
check("collapses runs of spaces/tabs", build_index.clean_remarks("a    b\t\tc") == "a b c")
check("strips ends", build_index.clean_remarks("  \r\n hi there \n ") == "hi there")
check("single spaces left alone", build_index.clean_remarks("one two") == "one two")

# ---------------------------------------------------------------- 3. header builder
full = build_index.build_header(listing(1))
check("full header", full == "SingleFamilyResidence | Irvine | 3 bd | 2.5 ba | 1,800 sqft | built 1990 | $900,000")
sparse = build_index.build_header(listing(1, L_Type_=None, L_City=None, L_Keyword2=None, LM_Dec_3=None,
                                          LM_Int2_3=None, YearBuilt=None, L_SystemPrice=None))
check("all-NULL header is empty", sparse == "")
check("NULL fields omitted, rest kept",
      build_index.build_header(listing(1, L_City=None, YearBuilt=None)) ==
      "SingleFamilyResidence | 3 bd | 2.5 ba | 1,800 sqft | $900,000")
check("no 'None' in any header", "None" not in build_index.build_header(listing(1, L_City=None, LM_Dec_3=None)))
zeros = build_index.build_header(listing(1, LM_Int2_3=0, YearBuilt=0, L_SystemPrice=0))
check("0 sqft/year/price treated as unknown", "sqft" not in zeros and "built" not in zeros and "$" not in zeros)
check("bedroom count of 0 is kept", "0 bd" in build_index.build_header(listing(1, L_Keyword2=0)))
check("whole-number baths have no .0", "2 ba" in build_index.build_header(listing(1, LM_Dec_3=Decimal("2.0"))))
check("blank city/type strings omitted", build_index.build_header(listing(1, L_City="", L_Type_=None)).startswith("3 bd"))
check("remarks come last", build_index.embedding_text(listing(1)).endswith(build_index.clean_remarks(LONG)))
check("header then remarks, one line", "\n" not in build_index.embedding_text(listing(1, "a\r\r\nb" + " c" * 50)))
check("empty header -> remarks only, no stray '.'",
      build_index.embedding_text(listing(1, L_Type_=None, L_City=None, L_Keyword2=None, LM_Dec_3=None,
                                         LM_Int2_3=None, YearBuilt=None, L_SystemPrice=None))
      == build_index.clean_remarks(LONG))

# ---------------------------------------------------------------- 4/5. search ranking and top_k
# Unit-length rows; the query equals row 30, so expected scores are known by hand.
MATRIX = np.array([[1, 0, 0], [0, 1, 0], [0.6, 0.8, 0], [0, 0, 1], [0.8, 0.6, 0]], dtype=np.float32)
IDS = [10, 20, 30, 40, 50]
META = {"model": build_index.MODEL_NAME, "dimension": 3, "row_count": 5, "min_remarks_chars": 80, "complete": True}
QVEC = [0.6, 0.8, 0.0]  # cosine with each row: 0.6, 0.8, 1.0, 0.0, 0.96


class SearchDB:
    """Mocks store.query: returns detail rows for the ids asked for, in scrambled order."""

    def __init__(self):
        self.calls = []

    def __call__(self, sql, params=None):
        self.calls.append((sql, list(params or [])))
        return [full_row(id=i, L_Address=f"{i} Main St") for i in reversed(params)]


def run_search(text, top_k=5, matrix=MATRIX, ids=IDS, meta=META, model=None, db=None):
    model = model or FakeModel(3, {"charming home": QVEC})
    db = db or SearchDB()
    with mock.patch.object(store, "load_index", return_value=(matrix, ids, meta)), \
         mock.patch.object(store, "_model", return_value=model), \
         mock.patch.object(store, "query", db):
        return store.search(text, top_k), model, db


res, model, db = run_search("charming home", 5)
check("ranked descending by similarity", [r["id"] for r in res] == [30, 50, 20, 10, 40])
check("scores are correct cosines", np.allclose([r["similarity"] for r in res], [1.0, 0.96, 0.8, 0.6, 0.0], atol=1e-6))
check("scores are monotonically non-increasing", all(a["similarity"] >= b["similarity"] for a, b in zip(res, res[1:])))
check("query encoded with normalize_embeddings=True", model.normalize_flags == [True])
check("score is a plain float", all(type(r["similarity"]) is float for r in res))
check("details hydrated for the results", res[0]["L_Address"] == "30 Main St")
check("hydration is one query", len(db.calls) == 1)
check("hydration only asks for the top_k ids", sorted(db.calls[0][1]) == [10, 20, 30, 40, 50])

res, _, db = run_search("charming home", 2)
check("top_k=2 returns 2", [r["id"] for r in res] == [30, 50])
check("hydrates only top_k ids", sorted(db.calls[0][1]) == [30, 50])
res, _, _ = run_search("charming home", 1)
check("top_k=1 returns best match", [r["id"] for r in res] == [30])
res, _, _ = run_search("charming home", 50)
check("top_k greater than rows returns all rows", len(res) == 5)
res, _, _ = run_search("charming home", 0)
check("top_k=0 is clamped to 1", len(res) == 1)
res, _, _ = run_search("charming home", 10_000)
check("huge top_k clamped without error", len(res) == 5)

class MissingRowDB(SearchDB):
    def __call__(self, sql, params=None):
        return [r for r in super().__call__(sql, params) if r["id"] != 30]

res, _, _ = run_search("charming home", 3, db=MissingRowDB())
check("listing gone from the table is skipped", [r["id"] for r in res] == [50, 20])

cleaned_model = FakeModel(3, {"charming home": QVEC})
run_search("  charming\r\n   home ", 1, model=cleaned_model)
check("query cleaned like remarks before encoding", cleaned_model.encoded == ["charming home"])

# ---------------------------------------------------------------- 6/7. empty query
for bad in ("", "   ", "\r\n\t  ", None):
    guard_model, guard_db = FakeModel(3), SearchDB()
    msg = raises(ValueError, run_search, bad, 5, model=guard_model, db=guard_db)
    check(f"empty query {bad!r} rejected", msg is not None and "empty" in msg.lower())
    check(f"empty query {bad!r} never reaches model or DB", not guard_model.encoded and not guard_db.calls)

# ---------------------------------------------------------------- 8. load_index validation
def write_index(directory, vectors=MATRIX, ids=IDS, meta=META):
    directory = Path(directory)
    np.save(directory / "vectors.npy", vectors)
    (directory / "ids.json").write_text(json.dumps(ids))
    (directory / "meta.json").write_text(json.dumps(meta))


def load_from(directory):
    store.load_index.cache_clear()
    with mock.patch.object(store, "OUT_DIR", Path(directory)):
        return store.load_index()


with tempfile.TemporaryDirectory() as tmp:
    write_index(tmp)
    v, i, m = load_from(tmp)
    check("valid index loads", len(i) == 5 and v.shape == (5, 3) and m["model"] == build_index.MODEL_NAME)
    for f in Path(tmp).iterdir():
        f.unlink()
    with mock.patch.object(store, "OUT_DIR", Path(tmp)):
        again = store.load_index()
    check("load_index is cached (no re-read)", again[1] is i)
    store.load_index.cache_clear()

with tempfile.TemporaryDirectory() as tmp:
    write_index(tmp, meta={**META, "model": "some-other-model"})
    msg = raises(ValueError, load_from, tmp)
    check("model-name mismatch raises", msg is not None and "some-other-model" in msg and build_index.MODEL_NAME in msg)

with tempfile.TemporaryDirectory() as tmp:
    write_index(tmp, ids=[10, 20, 30])
    msg = raises(ValueError, load_from, tmp)
    check("ids/vectors length disagreement raises", msg is not None and "disagree" in msg)

with tempfile.TemporaryDirectory() as tmp:
    write_index(tmp, meta={**META, "dimension": 384})
    msg = raises(ValueError, load_from, tmp)
    check("meta/vectors dimension disagreement raises", msg is not None and "disagree" in msg)

msg = raises(ValueError, run_search, "charming home", 5, model=FakeModel(dim=4, vectors={"charming home": [1, 0, 0, 0]}))
check("model/index dimension mismatch raises", msg is not None and "dimension" in msg)

for missing in ("vectors.npy", "ids.json", "meta.json"):
    with tempfile.TemporaryDirectory() as tmp:
        write_index(tmp)
        (Path(tmp) / missing).unlink()
        msg = raises(FileNotFoundError, load_from, tmp)
        check(f"missing {missing} raises clear error", msg is not None and missing in msg and "build_index.py" in msg)
with tempfile.TemporaryDirectory() as tmp:
    msg = raises(FileNotFoundError, load_from, tmp)
    check("empty index directory raises clear error", msg is not None and "build_index.py" in msg)
store.load_index.cache_clear()

# ---------------------------------------------------------------- 9. build: unit vectors, resume, checkpoint
def run_build(tmp, rows, limit=100, model=None, **patches):
    """Run build_index.main() against a temp output dir, a fake DB and a fake model."""
    model = model or FakeModel(3)
    with mock.patch.object(build_index, "OUT_DIR", Path(tmp)), \
         mock.patch.object(build_index, "query", FakeDB(rows)), \
         mock.patch.dict(sys.modules, {"sentence_transformers": fake_sentence_transformers(model)}), \
         mock.patch.object(sys, "argv", ["build_index.py", "--limit", str(limit)]):
        for name, value in patches.items():
            mock.patch.object(build_index, name, value).start()
        try:
            quiet(build_index.main)
        finally:
            mock.patch.stopall()
    return model


def read_out(tmp):
    d = Path(tmp)
    return (np.load(d / "vectors.npy"), json.loads((d / "ids.json").read_text()),
            json.loads((d / "meta.json").read_text()))


BUILD_ROWS = [listing(i, LONG + f"unique {i}") for i in range(1, 7)] + [listing(99, "too short")]

with tempfile.TemporaryDirectory() as tmp:
    model = run_build(tmp, BUILD_ROWS)
    vectors, ids, meta = read_out(tmp)
    check("build: short remarks left out of the index", ids == [1, 2, 3, 4, 5, 6])
    check("build: vectors are float32", vectors.dtype == np.float32)
    check("build: stored vectors are unit-normalized", np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5))
    check("build: encode called with normalize_embeddings=True", model.normalize_flags and all(model.normalize_flags))
    check("build: ids parallel to vectors", len(ids) == len(vectors) == meta["row_count"])
    check("build: meta records model", meta["model"] == build_index.MODEL_NAME)
    check("build: meta records dimension", meta["dimension"] == vectors.shape[1] == 3)
    check("build: meta records the length filter", meta["min_remarks_chars"] == 80)
    check("build: meta has build timestamp", "T" in meta["built_at"])
    check("build: meta marked complete", meta["complete"] is True)
    check("build: embedded text is header then remarks", model.encoded[0].startswith("SingleFamilyResidence | Irvine"))

    # resume: same index, two new listings appear -> only those are encoded
    more = BUILD_ROWS + [listing(7, LONG + "unique 7"), listing(8, LONG + "unique 8")]
    model2 = run_build(tmp, more)
    vectors2, ids2, meta2 = read_out(tmp)
    check("resume: skips ids already present", len(model2.encoded) == 2)
    check("resume: new ids appended after old ones", ids2 == [1, 2, 3, 4, 5, 6, 7, 8])
    check("resume: old vectors untouched", np.array_equal(vectors2[:6], vectors))
    check("resume: meta row_count updated", meta2["row_count"] == 8)

    model3 = run_build(tmp, more)
    check("resume: nothing to do when up to date", model3.encoded == [])

with tempfile.TemporaryDirectory() as tmp:
    run_build(tmp, BUILD_ROWS)
    meta_path = Path(tmp) / "meta.json"
    good = json.loads(meta_path.read_text())

    meta_path.write_text(json.dumps({**good, "model": "different-model"}))
    try:
        run_build(tmp, BUILD_ROWS + [listing(7, LONG + "unique 7")])
        refused = False
    except SystemExit as e:
        refused = "different settings" in str(e)
    check("resume refused when the model differs", refused)

    meta_path.write_text(json.dumps({**good, "min_remarks_chars": 40}))
    try:
        run_build(tmp, BUILD_ROWS + [listing(7, LONG + "unique 7")])
        refused = False
    except SystemExit as e:
        refused = "different settings" in str(e)
    check("resume refused when the length filter differs", refused)

    meta_path.write_text(json.dumps(good))
    (Path(tmp) / "ids.json").write_text(json.dumps([1, 2, 3]))
    try:
        run_build(tmp, BUILD_ROWS + [listing(7, LONG + "unique 7")])
        refused = False
    except SystemExit as e:
        refused = "inconsistent" in str(e)
    check("resume refused when ids/vectors disagree", refused)

with tempfile.TemporaryDirectory() as tmp:
    # Crash on the 3rd batch of 2 -> checkpoint after batch 2 (4 rows) should be on disk, then resume finishes.
    crashing = FakeModel(3, fail_after=4)
    try:
        run_build(tmp, BUILD_ROWS, model=crashing, CHECKPOINT_EVERY=2, BATCH_SIZE=2)
        crashed = False
    except RuntimeError:
        crashed = True
    check("checkpoint: simulated crash happened", crashed)
    vectors, ids, meta = read_out(tmp)
    check("checkpoint: partial index saved before the crash", ids == [1, 2, 3, 4] and len(vectors) == 4)
    check("checkpoint: partial index marked incomplete", meta["complete"] is False)
    resumed = run_build(tmp, BUILD_ROWS, model=FakeModel(3))
    vectors, ids, meta = read_out(tmp)
    check("checkpoint: restart finishes the remaining rows", ids == [1, 2, 3, 4, 5, 6] and meta["complete"] is True)
    check("checkpoint: restart encodes only what was missing", len(resumed.encoded) == 2)
    check("checkpoint: no leftover .tmp files", not list(Path(tmp).glob("*.tmp")))

# ---------------------------------------------------------------- 10. --json output
rows_for_cli = [full_row(id=1, LM_Dec_3=Decimal("2.5"), L_SystemPrice=Decimal("750000"),
                         L_Remarks="Bright\r\r\n\r\r\ncorner   unit. " + "word " * 80, similarity=0.7)]
out = io.StringIO()
with mock.patch.object(cli, "search", return_value=rows_for_cli), \
     mock.patch.object(sys, "argv", ["cli.py", "bright unit", "--json", "--top-k", "3"]), \
     contextlib.redirect_stdout(out):
    cli.main()
parsed = json.loads(out.getvalue())
check("--json output parses", isinstance(parsed, list) and len(parsed) == 1)
check("--json converts Decimal baths to a number", parsed[0]["LM_Dec_3"] == 2.5 and isinstance(parsed[0]["LM_Dec_3"], float))
check("--json converts Decimal price", parsed[0]["L_SystemPrice"] == 750000.0)
check("--json keeps similarity", parsed[0]["similarity"] == 0.7)
check("--json drops full remarks", "L_Remarks" not in parsed[0])
check("--json snippet capped at 150 chars", len(parsed[0]["remarks_snippet"]) == 150)
check("--json snippet is whitespace-cleaned", parsed[0]["remarks_snippet"].startswith("Bright corner unit. word"))
check("json_default handles dates", json.dumps({"d": __import__("datetime").date(2026, 1, 2)}, default=cli.json_default) == '{"d": "2026-01-02"}')

out = io.StringIO()
with mock.patch.object(cli, "search", return_value=rows_for_cli) as fake_search, \
     mock.patch.object(sys, "argv", ["cli.py", "bright unit", "--top-k", "3"]), \
     contextlib.redirect_stdout(out):
    cli.main()
check("--top-k passed through to search", fake_search.call_args[0] == ("bright unit", 3))
check("text output has score and snippet", "[0.700]" in out.getvalue() and "Bright corner unit." in out.getvalue())
check("text output snippet is cleaned (no CR/LF inside)", "\r" not in out.getvalue())

# ---------------------------------------------------------------- 11. NULL address rendering
check("full address", format_address(full_row()) == "1 Main St, Irvine, CA 92618")
check("NULL address omitted", format_address(full_row(L_Address=None)) == "Irvine, CA 92618")
check("NULL city omitted", format_address(full_row(L_City=None)) == "1 Main St, CA 92618")
check("NULL state omitted", format_address(full_row(L_State=None)) == "1 Main St, Irvine, 92618")
check("NULL zip omitted", format_address(full_row(L_Zip=None)) == "1 Main St, Irvine, CA")
check("all parts NULL -> placeholder", format_address(full_row(L_Address=None, L_City=None, L_State=None, L_Zip=None)) == "Address n/a")
check("empty-string parts omitted too", format_address(full_row(L_Address="", L_Zip="")) == "Irvine, CA")
every_null = [dict(L_Address=None), dict(L_City=None), dict(L_State=None), dict(L_Zip=None),
              dict(L_Address=None, L_City=None, L_State=None, L_Zip=None)]
check("cli result renders without 'None' for any NULL part",
      all("None" not in cli.format_result(1, full_row(**o)) for o in every_null))
check("cli result renders with NULL remarks", "None" not in cli.format_result(1, full_row(L_Remarks=None)))
check("property card renders without 'None' address", "None" not in format_property_card(full_row(L_Address=None, L_Zip=None)).split("\n")[0])
check("property card still shows the city", "Irvine" in format_property_card(full_row(L_Address=None)))

print()
print(f"{len(failures)} failure(s): {', '.join(failures)}" if failures else "All checks passed")
sys.exit(1 if failures else 0)
