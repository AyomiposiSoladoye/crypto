from datetime import datetime, timedelta, timezone

import pandas as pd

from analysis.classify import label_chat, primary_topic
from analysis.insights import _proper_nouns, label_announcements
from analysis.load import split

T0 = datetime(2025, 1, 1, 9, tzinfo=timezone.utc)


def frame(rows):
    df = pd.DataFrame(rows)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    df["local_time"] = df["published_at"].dt.tz_convert("Africa/Lagos")
    for c, d in [("fwd_channel_id", pd.NA), ("fwd_channel_post", pd.NA), ("views", None),
                 ("media_type", "text"), ("character_count", 10), ("link_count", 0),
                 ("forwards", None), ("reactions_total", None), ("poll_voters", None), ("is_forward", 0)]:
        if c not in df:
            df[c] = d
    df["message_id"] = [f"m{i}" for i in range(len(df))]
    return df


def test_topics():
    assert primary_topic("I can't login to the new app") == "app_access"
    assert primary_topic("my withdrawal is still pending since yesterday") == "deposit_withdrawal"
    assert primary_topic("funds on hold for 2 days") == "deposit_withdrawal"
    assert primary_topic("this is a scam link, don't click") == "scam_security"
    assert primary_topic("BTC going to 100k") == "trading_market"
    assert primary_topic("Good morning") == "greeting_social"


def test_kinds():
    df = frame([
        {"published_at": T0, "message_text": "/p btc"},
        {"published_at": T0, "message_text": "Bitcoin (BTC) Price: $70,963 USD 24hr Change: 5%"},
        {"published_at": T0, "message_text": "We apologise, please send me a DM"},
        {"published_at": T0, "message_text": "How do I withdraw naira?"},
    ])
    out = label_chat(df)
    assert out["kind"].tolist() == ["bot_command", "automated", "likely_staff", "member"]
    assert out["is_question"].tolist() == [False, False, False, True]
    assert out.loc[3, "topic"] == "deposit_withdrawal" and pd.isna(out.loc[0, "topic"])


def test_split_keeps_main_channel_and_dedupes_forwards():
    rows = [{"published_at": T0 + timedelta(hours=i), "message_text": f"post {i % 3}", "views": 100 + i % 3,
             "fwd_channel_id": 1, "fwd_channel_post": i % 3, "is_forward": 1} for i in range(6)]
    rows.append({"published_at": T0, "message_text": "spam", "views": 9999,
                 "fwd_channel_id": 2, "fwd_channel_post": 1, "is_forward": 1})
    rows.append({"published_at": T0, "message_text": "hello all", "is_forward": 0})
    ann, chat = split(frame(rows))
    assert len(ann) == 3 and ann["times_forwarded"].tolist() == [2, 2, 2]
    assert "spam" not in ann["message_text"].tolist()
    assert chat["message_text"].tolist() == ["hello all"]


def test_baseline_uses_only_earlier_posts():
    ann = pd.DataFrame({
        "first_forwarded_at": pd.date_range("2025-01-01", periods=40, freq="D", tz="UTC"),
        "views": [100.0] * 39 + [1000.0], "reactions_total": [1.0] * 40,
    })
    out = label_announcements(ann)
    assert out["baseline_views"].iloc[-1] == 100       # the spike doesn't raise its own baseline
    assert out["relative_views"].iloc[-1] == 10
    assert out["above_baseline"].iloc[:10].isna().all()  # too little history to judge


def test_names_are_detected_as_proper_nouns():
    texts = pd.Series(["thanks a lot Jameson for this", "ask Jameson about it", "yes Jameson said so",
                       "the migration is done", "after the migration it works", "why migration again"])
    names = _proper_nouns(texts)
    assert "jameson" in names and "migration" not in names
