"""SQLite storage for Telegram publications and their repeated measurements."""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    message_id        TEXT PRIMARY KEY,   -- telegram:<channel>:<id>
    platform          TEXT NOT NULL,
    channel           TEXT NOT NULL,
    telegram_id       INTEGER NOT NULL,
    published_at      TEXT NOT NULL,      -- UTC ISO-8601
    edited_at         TEXT,
    message_text      TEXT,
    media_type        TEXT,
    grouped_id        INTEGER,            -- album: several messages, one post
    pinned_flag       INTEGER,
    is_forward        INTEGER,
    character_count   INTEGER,
    word_count        INTEGER,
    link_count        INTEGER,
    poll_question     TEXT,
    poll_option_count INTEGER,
    poll_is_quiz      INTEGER,
    content_pillar    TEXT,               -- filled in later by classification/review
    first_seen_at     TEXT NOT NULL,
    last_seen_at      TEXT NOT NULL
);

-- One row per observation. Counters are cumulative at collected_at.
CREATE TABLE IF NOT EXISTS measurements (
    message_id      TEXT NOT NULL REFERENCES messages(message_id),
    collected_at    TEXT NOT NULL,
    age_hours       REAL NOT NULL,
    views           INTEGER,
    forwards        INTEGER,
    reactions_total INTEGER,
    reactions_json  TEXT,
    comments        INTEGER,
    poll_voters     INTEGER,
    PRIMARY KEY (message_id, collected_at)
);

CREATE TABLE IF NOT EXISTS channel_snapshots (
    channel      TEXT NOT NULL,
    collected_at TEXT NOT NULL,
    subscribers  INTEGER,
    PRIMARY KEY (channel, collected_at)
);
"""

MESSAGE_COLUMNS = [
    "message_id", "platform", "channel", "telegram_id", "published_at", "edited_at",
    "message_text", "media_type", "grouped_id", "pinned_flag", "is_forward",
    "character_count", "word_count", "link_count", "poll_question",
    "poll_option_count", "poll_is_quiz",
]
MEASUREMENT_COLUMNS = [
    "message_id", "collected_at", "age_hours", "views", "forwards",
    "reactions_total", "reactions_json", "comments", "poll_voters",
]


def connect(path: str | Path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def upsert_message(conn: sqlite3.Connection, record: dict, seen_at: str) -> None:
    cols = ", ".join(MESSAGE_COLUMNS)
    placeholders = ", ".join("?" for _ in MESSAGE_COLUMNS)
    updates = ", ".join(f"{c}=excluded.{c}" for c in MESSAGE_COLUMNS if c != "message_id")
    conn.execute(
        f"""INSERT INTO messages ({cols}, first_seen_at, last_seen_at)
            VALUES ({placeholders}, ?, ?)
            ON CONFLICT(message_id) DO UPDATE SET {updates}, last_seen_at=excluded.last_seen_at""",
        [record[c] for c in MESSAGE_COLUMNS] + [seen_at, seen_at],
    )


def insert_measurement(conn: sqlite3.Connection, m: dict) -> None:
    cols = ", ".join(MEASUREMENT_COLUMNS)
    placeholders = ", ".join("?" for _ in MEASUREMENT_COLUMNS)
    conn.execute(
        f"INSERT OR IGNORE INTO measurements ({cols}) VALUES ({placeholders})",
        [m[c] for c in MEASUREMENT_COLUMNS],
    )


def insert_channel_snapshot(conn, channel: str, collected_at: str, subscribers: int | None) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO channel_snapshots VALUES (?, ?, ?)",
        (channel, collected_at, subscribers),
    )


def recent_telegram_ids(conn, channel: str, since_iso: str) -> list[int]:
    rows = conn.execute(
        "SELECT telegram_id FROM messages WHERE channel=? AND published_at>=? ORDER BY telegram_id",
        (channel, since_iso),
    )
    return [r[0] for r in rows]
