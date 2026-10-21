"""
Week 4 tests for session memory and slot filling. Zero Gemini calls and zero DB calls:
filters are fed in as dicts, and the parser and search are replaced with fakes.

Usage:
    python tests/test_session.py
"""

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chat import converse  # noqa: E402
from session import session_store as store  # noqa: E402
from session.session_store import (  # noqa: E402
    clear_session, get_session, merge_filters, new_session, update_session,
)

# Keep test sessions out of the real .sessions/ folder.
store.SESSION_DIR = Path(tempfile.mkdtemp())

failures = []


def check(name, condition):
    print(("PASS" if condition else "FAIL"), name)
    if not condition:
        failures.append(name)


def filters(**kw):
    return {f: kw.get(f) for f in store.FILTER_FIELDS}


# --- merge_filters ---
base = {**new_session(), "city": "Irvine", "max_price": 1500000, "pool": True}
merged = merge_filters(base, filters(min_beds=3))
check("merge keeps old values when new is null", merged["city"] == "Irvine" and merged["max_price"] == 1500000)
check("merge adds new non-null values", merged["min_beds"] == 3)

merged = merge_filters(base, filters(city="Tustin", max_price=900000))
check("merge overwrites when new is non-null", merged["city"] == "Tustin" and merged["max_price"] == 900000)

merged = merge_filters(base, filters(pool=False))
check("pool=False overwrites True", merged["pool"] is False)
merged = merge_filters(merged, filters(min_beds=2))
check("pool=False survives a later merge", merged["pool"] is False)
check("merge does not mutate the input session", base["pool"] is True)

# --- persistence and TTL ---
update_session("u1", {"city": "Irvine"})
check("session persists to disk", get_session("u1")["city"] == "Irvine")

path = store.session_path("u1")
path.write_text(path.read_text().replace(str(get_session("u1")["updated_at"]), str(time.time() - 61 * 60)))
check("session older than 60 min resets on read", get_session("u1")["city"] is None)
check("expired session file is removed", not path.exists())

update_session("u2", {"city": "Irvine"})
check("fresh session is kept", get_session("u2")["city"] == "Irvine")

# --- slot filling (pure logic) ---
s = new_session()
check("no city -> ask city", converse.next_step(s) == "ask_city")
s["city"] = "Irvine"
check("city but no price -> ask price", converse.next_step(s) == "ask_price")
s["max_price"] = 1000000
check("city + price -> ask one follow-up", converse.next_step(s) == "ask_followup")
s["asked_followup"] = True
check("follow-up already asked -> search", converse.next_step(s) == "search")
s = {**new_session(), "city": "Irvine", "max_price": 1000000, "min_beds": 3}
check("beds already given -> no follow-up, search", converse.next_step(s) == "search")
s = {**new_session(), "conversation_step": 2}
check("question cap reached -> no more questions", converse.next_step(s) == "need_city")
s = {**new_session(), "city": "Irvine", "max_price": 1000000, "conversation_step": 2}
check("question cap reached -> search without follow-up", converse.next_step(s) == "search")

# --- full turns with fake parser + fake search ---
searches = []


def fake_search(f, page, limit):
    searches.append((dict(f), page))
    return [{
        "id": page * 10 + i, "L_Address": f"{i} Main St", "L_City": f["city"], "L_State": "CA",
        "L_Zip": "92618", "L_Type_": "Condominium", "L_SystemPrice": 800000, "L_Keyword2": 3,
        "LM_Dec_3": 2, "LM_Int2_3": 1500, "AssociationFee": None, "DaysOnMarket": 5,
        "PoolPrivateYN": "", "ViewYN": "", "PhotoCount": 10,
    } for i in range(limit)]


def no_parser(message):
    raise AssertionError("parser must not be called")


converse.search_active_listings = fake_search

# Each scripted reply is what the parser would have returned for that message.
script = iter([filters(city="Irvine"), filters(max_price=1200000), filters()])
parser_calls = []


def scripted_parser(message):
    parser_calls.append(message)
    return next(script)


r1 = converse.handle_turn("t1", "homes in Irvine", scripted_parser)
r2 = converse.handle_turn("t1", "1.2M", scripted_parser)
check("asks for price after city", "maximum price" in r1)
check("asks follow-up once both known", "property type" in r2 and not searches)
r3 = converse.handle_turn("t1", "no preference", scripted_parser)
check("searches after follow-up answered", len(searches) == 1 and "Page 1" in r3)
check("search used merged filters", searches[0][0]["city"] == "Irvine" and searches[0][0]["max_price"] == 1200000)
check("one parser call per message", len(parser_calls) == 3)
check("at most 2 questions asked", get_session("t1")["conversation_step"] == 2)

# Pagination: no parser call, page advances on stored filters.
r4 = converse.handle_turn("t1", "More!", no_parser)
check("'more' returns page 2 without parsing", "Page 2" in r4 and searches[-1][1] == 2)
check("page 2 used the stored filters", searches[-1][0]["city"] == "Irvine")
check("last_page stored", get_session("t1")["last_page"] == 2)
check("last_results stored", len(get_session("t1")["last_results"]) == converse.PAGE_SIZE)
converse.handle_turn("t1", "show me more", no_parser)
check("'show me more' advances to page 3", get_session("t1")["last_page"] == 3)

# Reset clears everything, with no parser call.
converse.handle_turn("t1", "start over", no_parser)
check("reset clears the session", get_session("t1")["city"] is None and get_session("t1")["last_page"] == 0)
check("'more' with no search yet is handled", "no search yet" in converse.handle_turn("t1", "next", no_parser))

# Both slots in one message: still at most one follow-up, then search.
n = len(searches)
reply = converse.handle_turn("t2", "condos in Tustin under 900k", lambda m: filters(city="Tustin", max_price=900000, property_type="Condominium"))
check("type given up front -> searches immediately", len(searches) == n + 1)

# Hitting the cap with nothing useful: statement, not a third question.
junk = lambda m: filters()
converse.handle_turn("t3", "hi", junk)
converse.handle_turn("t3", "hello?", junk)
reply = converse.handle_turn("t3", "uh", junk)
check("never asks a 3rd question", "?" not in reply and get_session("t3")["conversation_step"] == 2)

clear_session("u2")
if failures:
    print(f"\n{len(failures)} FAILED")
    sys.exit(1)
print("\nAll passed")
