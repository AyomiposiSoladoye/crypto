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
