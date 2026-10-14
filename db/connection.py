"""
Week 3: MySQL connection helper.

Reads MYSQL_HOST / MYSQL_USER / MYSQL_PASSWORD / MYSQL_DATABASE from the repo's .env
and exposes one function, query(), that every other module uses to talk to the database.

Always pass user-supplied values through `params` (as %s placeholders), never by
building the SQL string yourself. The driver escapes them, which blocks SQL injection.
"""

import os
from pathlib import Path

import mysql.connector
from dotenv import load_dotenv

# .env lives at the repo root, one level above this db/ folder.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def query(sql, params=None):
    """Run one SQL statement and return the rows as a list of dicts."""
    conn = mysql.connector.connect(
        host=os.environ["MYSQL_HOST"],
        user=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASSWORD"],
        database=os.environ["MYSQL_DATABASE"],
    )
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, params or ())
        return cur.fetchall()
    finally:
        conn.close()  # always release the connection, even if the query failed
