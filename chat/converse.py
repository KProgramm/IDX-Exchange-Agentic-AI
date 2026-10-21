"""
Week 4: multi-turn conversational search with session memory.

Each call to handle_turn() is one user message. Everything we need to remember
(filters, last page, how many questions we've asked) is saved by session_store.

Flow for each message:
1. "more" / "reset" style commands are matched with plain string checks (no Gemini).
2. Anything else gets exactly ONE parser call and is merged into the session.
3. If city or max_price is still missing, ask for it (one question per turn).
4. Once both are known, ask at most one follow-up about type / beds, then search.
   We never ask more than MAX_QUESTIONS questions in total.

Usage:
    python chat/converse.py                 (interactive; --user-id NAME, default "cli")
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so the imports below work when run as a script
sys.path.insert(0, str(ROOT / "parser"))

from search.search_listings import format_property_card, search_active_listings  # noqa: E402
from session.session_store import (  # noqa: E402
    FILTER_FIELDS, clear_session, get_session, merge_filters, update_session,
)

PAGE_SIZE = 5
MAX_QUESTIONS = 2

NEXT_COMMANDS = {"more", "next", "show me more", "next page", "show more"}
RESET_COMMANDS = {"reset", "start over", "new search"}


def normalize(message):
    """Lowercase and strip punctuation so 'More!' still matches 'more'."""
    return re.sub(r"[^a-z ]", "", message.lower()).strip()


def parse(message):
    # Imported here so navigation commands never need Gemini installed or configured.
    from parse_query import parse_property_query
    return parse_property_query(message)


def next_step(session):
    """Decide what to do with the current session: ask_city, ask_price, ask_followup, or search."""
    asked = session["conversation_step"]

    if session["city"] is None:
        return "ask_city" if asked < MAX_QUESTIONS else "need_city"
    if session["max_price"] is None:
        return "ask_price" if asked < MAX_QUESTIONS else "need_price"

    no_detail = session["property_type"] is None and session["min_beds"] is None
    if no_detail and not session["asked_followup"] and asked < MAX_QUESTIONS:
        return "ask_followup"
    return "search"


QUESTIONS = {
    "ask_city": "Which city would you like to search in?",
    "ask_price": "What's your maximum price?",
    "ask_followup": "Any preference on property type (house, condo, townhouse) or number of bedrooms?",
    # Out of questions: say what's needed as a statement, not another question.
    "need_city": "I still need a city to search. Tell me one, like \"Irvine\".",
    "need_price": "I still need a maximum price to search. Tell me one, like \"$1.2M\".",
}


def show_page(user_id, session, page):
    """Run the search for `page`, save the results, and return the cards as text."""
    filters = {field: session[field] for field in FILTER_FIELDS}
    rows = search_active_listings(filters, page, PAGE_SIZE)

    if not rows:
        if page > 1:
            return "That's all the listings I have for this search. Say \"reset\" to start a new one."
        return "No active listings match those filters. Try a higher price or fewer requirements."

    update_session(user_id, {"last_results": rows, "last_page": page})
    cards = "\n\n".join(format_property_card(r) for r in rows)
    return f"Page {page}: {len(rows)} listings\n\n{cards}\n\nSay \"more\" for the next page."


def handle_turn(user_id, message, parser=parse):
    """Process one user message and return the reply text."""
    command = normalize(message)

    if command in RESET_COMMANDS:
        clear_session(user_id)
        return "Okay, starting over. What are you looking for?"

    session = get_session(user_id)

    if command in NEXT_COMMANDS:
        if session["last_page"] == 0:
            return "There's no search yet. Tell me a city and a budget to get started."
        return show_page(user_id, session, session["last_page"] + 1)

    # One parser call per message. Merge, then ask or search.
    session = merge_filters(session, parser(message))
    step = next_step(session)
    filters = {field: session[field] for field in FILTER_FIELDS}

    if step == "search":
        update_session(user_id, {**filters, "asked_followup": True})
        return show_page(user_id, session, 1)

    updates = dict(filters)
    if step.startswith("ask_"):
        updates["conversation_step"] = session["conversation_step"] + 1
        updates["asked_followup"] = session["asked_followup"] or step == "ask_followup"
    update_session(user_id, updates)
    return QUESTIONS[step]


def main():
    ap = argparse.ArgumentParser(description="Chat with the listing search agent.")
    ap.add_argument("--user-id", default="cli")
    args = ap.parse_args()

    print('Real estate search. Type "reset" to start over, Ctrl+C to quit.\n')
    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if message:
            print(f"\n{handle_turn(args.user_id, message)}\n")


if __name__ == "__main__":
    main()
