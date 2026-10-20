"""Telegram channel collector.

    python -m ingest.telegram.scrape login
    python -m ingest.telegram.scrape backfill [--since 2024-01-01]
    python -m ingest.telegram.scrape snapshot [--days 8]
    python -m ingest.telegram.scrape export [--out data/telegram_messages.csv]

backfill  - every historical message once, with its lifetime counters as of today.
snapshot  - re-reads messages from the last N days and stores a new measurement row.
            Schedule it hourly so 1h / 6h / 24h / 7d windows can be reconstructed.
"""
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from telethon.errors import PhoneNumberInvalidError
from telethon.sync import TelegramClient
from telethon.tl.functions.channels import GetFullChannelRequest

from . import store
from .parse import message_to_measurement, message_to_record

BATCH = 100  # get_messages(ids=...) accepts up to 100 ids per request


def load_env(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def settings() -> dict:
    load_env()
    missing = [k for k in ("TG_API_ID", "TG_API_HASH", "TG_CHANNELS") if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"Missing in .env: {', '.join(missing)} (see .env.example)")
    return {
        "api_id": int(os.environ["TG_API_ID"]),
        "api_hash": os.environ["TG_API_HASH"],
        "session": os.environ.get("TG_SESSION", "secrets/telegram"),
        "channels": [_channel_ref(c) for c in os.environ["TG_CHANNELS"].split(",") if c.strip()],
        "db": os.environ.get("DB_PATH", "data/community.db"),
    }


def _channel_ref(value: str) -> str | int:
    """Accept 'name', '@name', 't.me/name' or a numeric id like -1001234567890 (private channels)."""
    value = value.strip().lstrip("@")
    for prefix in ("https://", "http://", "t.me/"):
        value = value.removeprefix(prefix)
    return int(value) if value.lstrip("-").isdigit() else value


def client_for(cfg: dict) -> TelegramClient:
    Path(cfg["session"]).parent.mkdir(parents=True, exist_ok=True)
    # Telethon sleeps automatically on FloodWait errors shorter than this many seconds.
    return TelegramClient(cfg["session"], cfg["api_id"], cfg["api_hash"], flood_sleep_threshold=120)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def snapshot_subscribers(client, conn, channel: str, collected_at: str) -> None:
    full = client(GetFullChannelRequest(channel))
    store.insert_channel_snapshot(conn, channel, collected_at, full.full_chat.participants_count)


def _ask_phone() -> str:
    return input("Phone number in international format, e.g. +2348012345678: ").strip().replace(" ", "")


def cmd_login(cfg: dict) -> None:
    client = client_for(cfg)
    try:
        client.start(phone=_ask_phone)  # then prompts for the code and 2FA password
        me = client.get_me()
        print(f"Logged in as id={me.id}. Session saved to {cfg['session']}.session")
    except PhoneNumberInvalidError:
        raise SystemExit(
            "Telegram rejected that number. Start with + and your country code and drop the "
            "leading 0, e.g. 08012345678 in Nigeria becomes +2348012345678. Then run login again."
        )
    finally:
        client.disconnect()


def cmd_backfill(cfg: dict, since: datetime | None) -> None:
    conn = store.connect(cfg["db"])
    with client_for(cfg) as client:
        for channel in cfg["channels"]:
            collected = now_utc()
            seen = collected.isoformat()
            snapshot_subscribers(client, conn, channel, seen)
            n = 0
            # reverse=True walks oldest -> newest; offset_date then acts as the start date.
            for msg in client.iter_messages(channel, reverse=True, offset_date=since, wait_time=1):
                if getattr(msg, "action", None) is not None:  # service message (pin, title change...)
                    continue
                store.upsert_message(conn, message_to_record(msg, channel), seen)
                store.insert_measurement(conn, message_to_measurement(msg, channel, collected))
                n += 1
                if n % 500 == 0:
                    conn.commit()
                    print(f"  {channel}: {n} messages…")
            conn.commit()
            print(f"{channel}: backfilled {n} messages")


def cmd_snapshot(cfg: dict, days: int) -> None:
    conn = store.connect(cfg["db"])
    with client_for(cfg) as client:
        for channel in cfg["channels"]:
            collected = now_utc()
            seen = collected.isoformat()
            snapshot_subscribers(client, conn, channel, seen)

            # Pick up anything posted since the last run.
            last = conn.execute(
                "SELECT MAX(telegram_id) FROM messages WHERE channel=?", (channel,)
            ).fetchone()[0] or 0
            for msg in client.iter_messages(channel, min_id=last, reverse=True):
                if getattr(msg, "action", None) is None:
                    store.upsert_message(conn, message_to_record(msg, channel), seen)

            ids = store.recent_telegram_ids(conn, channel, (collected - timedelta(days=days)).isoformat())
            n = 0
            for i in range(0, len(ids), BATCH):
                for msg in client.get_messages(channel, ids=ids[i:i + BATCH]):
                    if msg is None:  # deleted since we first saw it
                        continue
                    store.upsert_message(conn, message_to_record(msg, channel), seen)
                    store.insert_measurement(conn, message_to_measurement(msg, channel, collected))
                    n += 1
            conn.commit()
            print(f"{channel}: measured {n} recent messages")


def cmd_export(cfg: dict, out: str) -> None:
    """Flat CSV: one row per message with its most recent measurement."""
    conn = store.connect(cfg["db"])
    rows = conn.execute(
        """SELECT m.*, x.collected_at, x.age_hours, x.views, x.forwards,
                  x.reactions_total, x.reactions_json, x.comments, x.poll_voters
           FROM messages m
           JOIN measurements x ON x.message_id = m.message_id
           WHERE x.collected_at = (SELECT MAX(collected_at) FROM measurements
                                   WHERE message_id = m.message_id)
           ORDER BY m.channel, m.published_at"""
    )
    headers = [d[0] for d in rows.description]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:  # BOM so Excel shows emoji correctly
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("login")
    b = sub.add_parser("backfill")
    b.add_argument("--since", type=lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc))
    s = sub.add_parser("snapshot")
    s.add_argument("--days", type=int, default=8)
    e = sub.add_parser("export")
    e.add_argument("--out", default="data/telegram_messages.csv")
    args = parser.parse_args()

    cfg = settings()
    if args.cmd == "login":
        cmd_login(cfg)
    elif args.cmd == "backfill":
        cmd_backfill(cfg, args.since)
    elif args.cmd == "snapshot":
        cmd_snapshot(cfg, args.days)
    else:
        cmd_export(cfg, args.out)


if __name__ == "__main__":
    main()
