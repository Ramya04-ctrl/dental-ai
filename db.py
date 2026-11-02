"""
db.py  -  SQLite database for the screening history.

Privacy: we store ONLY a random reference id, the date, the results and the
image-quality numbers. We do NOT store names and we do NOT store the photos.
"""
import sqlite3
from contextlib import closing
from datetime import datetime

import pandas as pd

import config

COLUMNS = ["ref_id", "screened_at", "age_group", "condition", "confidence", "risk",
           "urgency", "blur_score", "brightness"]


def _connect():
    """Opens the database. 'closing' makes sure the connection is closed after use."""
    config.ensure_dirs()
    return closing(sqlite3.connect(config.DB_PATH))


def init_db():
    with _connect() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS screenings (
            ref_id TEXT PRIMARY KEY, screened_at TEXT, age_group TEXT, condition TEXT,
            confidence REAL, risk TEXT, urgency TEXT, blur_score REAL, brightness REAL)""")
        con.commit()


def save_screening(record: dict):
    init_db()
    record = dict(record)
    record.setdefault("screened_at", datetime.now().strftime("%Y-%m-%d %H:%M"))
    with _connect() as con:
        con.execute(f"INSERT OR REPLACE INTO screenings ({','.join(COLUMNS)}) "
                    f"VALUES ({','.join('?' * len(COLUMNS))})",
                    [record.get(c) for c in COLUMNS])
        con.commit()


def get_history(limit=200) -> pd.DataFrame:
    init_db()
    with _connect() as con:
        return pd.read_sql_query(
            "SELECT * FROM screenings ORDER BY screened_at DESC LIMIT ?", con, params=(limit,))
