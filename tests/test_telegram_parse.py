from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

from ingest.telegram import store
from ingest.telegram.parse import message_to_measurement, message_to_record


class MessageMediaPoll(NS):
    pass


class MessageMediaDocument(NS):
    pass


class ReactionEmoji(NS):
    pass


class ReactionCustomEmoji(NS):
    pass


PUBLISHED = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


def make_msg(**kw):
    base = dict(
        id=42, date=PUBLISHED, message="What is a seed phrase? Read more https://x.co/a",
        media=None, edit_date=None, grouped_id=None, pinned=False, fwd_from=None,
        views=1200, forwards=15, replies=None, reactions=None,
    )
    base.update(kw)
    return NS(**base)


def test_text_message_record():
    r = message_to_record(make_msg(), "exchange")
    assert r["message_id"] == "telegram:exchange:42"
    assert r["media_type"] == "text"
    assert r["link_count"] == 1
    assert r["published_at"] == "2026-09-01T12:00:00+00:00"


def test_missing_counters_stay_none_not_zero():
    m = message_to_measurement(make_msg(views=None, forwards=None), "exchange",
                               PUBLISHED + timedelta(hours=24))
    assert m["views"] is None and m["forwards"] is None
    assert m["reactions_total"] is None and m["comments"] is None
    assert m["age_hours"] == 24


def test_reactions_and_poll():
    poll = MessageMediaPoll(
        poll=NS(question=NS(text="Which coin?"), answers=[1, 2, 3], quiz=False),
        results=NS(total_voters=310),
    )
    reactions = NS(results=[
        NS(reaction=ReactionEmoji(emoticon="🔥"), count=40),
        NS(reaction=ReactionCustomEmoji(document_id=99), count=2),
    ])
    msg = make_msg(media=poll, reactions=reactions,
                   replies=NS(replies=7, comments=True))
    r = message_to_record(msg, "exchange")
    m = message_to_measurement(msg, "exchange", PUBLISHED + timedelta(hours=6))
    assert (r["media_type"], r["poll_question"], r["poll_option_count"]) == ("poll", "Which coin?", 3)
    assert m["poll_voters"] == 310
    assert m["reactions_total"] == 42
    assert "custom:99" in m["reactions_json"]
    assert m["comments"] == 7


def test_video_document():
    msg = make_msg(media=MessageMediaDocument(document=NS(mime_type="video/mp4")))
    assert message_to_record(msg, "exchange")["media_type"] == "video"


def test_store_keeps_repeated_measurements(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    msg = make_msg()
    seen = PUBLISHED.isoformat()
    for hours, views in [(1, 300), (24, 1200)]:
        store.upsert_message(conn, message_to_record(msg, "exchange"), seen)
        store.insert_measurement(conn, message_to_measurement(
            make_msg(views=views), "exchange", PUBLISHED + timedelta(hours=hours)))
    assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 1
    assert [r[0] for r in conn.execute("SELECT views FROM measurements ORDER BY age_hours")] == [300, 1200]


class PeerChannel(NS):
    pass


class PeerUser(NS):
    pass


def test_group_member_text_dropped_but_channel_forward_kept():
    member = make_msg(message="my phone is 0801...", fwd_from=None)
    forwarded = make_msg(message="Scheduled maintenance tonight",
                         fwd_from=NS(from_id=PeerChannel(channel_id=777), channel_post=55))
    from_user = make_msg(message="forwarded from a person",
                         fwd_from=NS(from_id=PeerUser(user_id=1), channel_post=None))
    assert message_to_record(member, "grp", keep_text=False)["message_text"] is None
    r = message_to_record(forwarded, "grp", keep_text=False)
    assert (r["message_text"], r["fwd_channel_id"], r["fwd_channel_post"]) == (
        "Scheduled maintenance tonight", 777, 55)
    u = message_to_record(from_user, "grp", keep_text=False)
    assert u["message_text"] is None and u["fwd_channel_id"] is None


def test_old_database_gets_new_columns(tmp_path):
    import sqlite3
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE messages (message_id TEXT PRIMARY KEY, channel TEXT)")
    old.commit(); old.close()
    conn = store.connect(path)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(messages)")}
    assert {"fwd_channel_id", "fwd_channel_post"} <= cols


def test_member_text_redacted():
    msg = make_msg(message="DM @quidax_helper or call 08012345678, send to "
                           "0x52908400098527886E0F7030069857D2E4169EE7 / me@mail.com acct 0123456789. "
                           "BTC hit 100000 today, 2 hours ago")
    t = message_to_record(msg, "grp", keep_text=False, member_text="redact")["message_text"]
    assert "08012345678" not in t and "0x5290" not in t and "me@mail.com" not in t
    assert "@quidax_helper" not in t and "0123456789" not in t
    assert "<WALLET>" in t and "<EMAIL>" in t and "<USER>" in t
    assert "100000" in t and "2 hours" in t   # ordinary numbers survive


def test_redaction_leaves_prices_and_links_alone():
    text = "Price: 0.000006277 ETH, see https://x.com/quidaxglobal/status/1834567890123456789?s=46"
    assert message_to_record(make_msg(message=text), "grp", keep_text=False,
                             member_text="redact")["message_text"] == text
