"""Load collected Telegram data and split it into announcements and member chat."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

LATEST_SQL = """
SELECT m.*, x.collected_at, x.age_hours, x.views, x.forwards,
       x.reactions_total, x.reactions_json, x.comments, x.poll_voters
FROM messages m
JOIN measurements x ON x.message_id = m.message_id
WHERE x.collected_at = (SELECT MAX(collected_at) FROM measurements WHERE message_id = m.message_id)
"""
LOCAL_TZ = "Africa/Lagos"

# Fallback for exports without fwd_channel_id: forwarded trading-signal spam (often Chinese-language).
SIGNAL_SPAM = re.compile(r"[\u4e00-\u9fff]|\bTP\s?1\b|\bSL\b.*\d|進場|止損", re.IGNORECASE)


def load_messages(source: str | Path) -> pd.DataFrame:
    """One row per message with its latest measurement, from community.db or an export CSV."""
    source = Path(source)
    if source.suffix == ".csv":
        df = pd.read_csv(source, encoding="utf-8-sig", low_memory=False)
    else:
        with sqlite3.connect(source) as conn:
            df = pd.read_sql_query(LATEST_SQL, conn)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df["local_time"] = df["published_at"].dt.tz_convert(LOCAL_TZ)
    for col in ("fwd_channel_id", "fwd_channel_post"):
        if col not in df:
            df[col] = pd.NA
    return df


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(announcements, chat).

    Announcements are posts forwarded into the group from its main source channel; they carry
    the original post's view count. Forwards from other channels (often spam) are left out.
    The same post is often forwarded several times, so keep one row per source post
    (by channel/post id, or text + views for older exports).
    """
    is_ann = df["views"].notna()
    ann = df[is_ann].copy()
    if ann["fwd_channel_id"].notna().any():
        # The group's own announcement channel is the one most posts are forwarded from.
        official = ann.groupby("fwd_channel_id")["fwd_channel_post"].nunique().idxmax()
        ann = ann[ann["fwd_channel_id"] == official]
    else:
        ann = ann[~ann["message_text"].fillna("").str.contains(SIGNAL_SPAM)]
    by_id = ann["fwd_channel_post"].notna()
    ann["post_key"] = ann["fwd_channel_id"].astype(str) + ":" + ann["fwd_channel_post"].astype(str)
    ann.loc[~by_id, "post_key"] = (
        ann.loc[~by_id, "message_text"].fillna("").str[:80] + "|" + ann.loc[~by_id, "views"].astype(int).astype(str)
    )
    ann = (
        ann.sort_values("published_at")
        .groupby("post_key", as_index=False)
        .agg(
            first_forwarded_at=("published_at", "first"),
            local_time=("local_time", "first"),
            times_forwarded=("message_id", "size"),
            message_text=("message_text", "first"),
            media_type=("media_type", "first"),
            character_count=("character_count", "first"),
            link_count=("link_count", "first"),
            views=("views", "max"),
            forwards=("forwards", "max"),
            reactions_total=("reactions_total", "max"),
            poll_voters=("poll_voters", "max"),
        )
        .sort_values("first_forwarded_at", ignore_index=True)
    )
    chat = df[~is_ann & (df["is_forward"] == 0)].copy()
    return ann, chat
