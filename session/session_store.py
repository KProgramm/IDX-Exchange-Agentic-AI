"""
Week 4: per-user search session, saved as JSON at .sessions/<user_id>.json.

It lives on disk (not in a Python variable) because OpenClaw runs each skill as a
fresh subprocess, so anything held in memory is gone by the next turn.

Functions:
    get_session(user_id)                -> session dict (fresh one if missing or expired)
    update_session(user_id, updates)    -> merge `updates` into the saved session
    clear_session(user_id)              -> delete the session
    merge_filters(session, new_filters) -> session with parser filters folded in
"""

import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SESSION_DIR = ROOT / ".sessions"
TTL_SECONDS = 60 * 60  # a session older than 60 minutes resets on read

FILTER_FIELDS = [
    "city", "max_price", "min_beds", "min_baths", "min_sqft",
    "property_type", "pool", "view", "max_hoa",
]


def new_session():
    return {
        **{field: None for field in FILTER_FIELDS},
        "last_results": [],
        "last_page": 0,         # 0 = no search has run yet
        "conversation_step": 0,  # how many questions we have asked so far
        "asked_followup": False,  # the optional beds/type question was already asked
        "updated_at": time.time(),
    }


def session_path(user_id):
    # Keep only safe characters so a user_id can never point outside .sessions/.
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(user_id)).lstrip(".") or "_"
    return SESSION_DIR / f"{safe}.json"


def get_session(user_id):
    """Load the saved session, or a fresh one if it's missing, unreadable, or expired."""
    path = session_path(user_id)
    try:
        session = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return new_session()

    if time.time() - session.get("updated_at", 0) > TTL_SECONDS:
        clear_session(user_id)
        return new_session()

    # Fill in any field an older session file is missing.
    return {**new_session(), **session}


def update_session(user_id, updates):
    """Apply `updates` to the session, stamp updated_at, save, and return it."""
    session = get_session(user_id)
    session.update(updates)
    session["updated_at"] = time.time()

    SESSION_DIR.mkdir(exist_ok=True)
    # default=str: DB rows can hold Decimal / date values that JSON can't store directly.
    session_path(user_id).write_text(json.dumps(session, indent=2, default=str))
    return session


def clear_session(user_id):
    session_path(user_id).unlink(missing_ok=True)


def merge_filters(session, new_filters):
    """Return a copy of `session` where non-null parser values overwrite old ones."""
    merged = dict(session)
    for field in FILTER_FIELDS:
        # "is not None", not truthiness: pool=False and view=False are real answers.
        if new_filters.get(field) is not None:
            merged[field] = new_filters[field]
    return merged
