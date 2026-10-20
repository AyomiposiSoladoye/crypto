"""Turn Telethon Message objects into plain records.

Kept free of network code so it can be unit-tested with fake objects.
Deliberately ignores member-identifying fields (from_id, post_author,
recent_reactions, top_reactors, recent_repliers, poll voters).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def _text_of(value) -> str | None:
    """Poll questions/answers are plain str in old layers, TextWithEntities in new ones."""
    if value is None:
        return None
    return getattr(value, "text", value)


def media_type(msg) -> str:
    media = getattr(msg, "media", None)
    if media is None:
        return "text"
    name = type(media).__name__
    if name == "MessageMediaPoll":
        return "poll"
    if name == "MessageMediaPhoto":
        return "image"
    if name == "MessageMediaWebPage":
        return "link"
    if name == "MessageMediaDocument":
        mime = getattr(getattr(media, "document", None), "mime_type", "") or ""
        if mime.startswith("video/"):
            return "video"
        if mime.startswith("audio/"):
            return "audio"
        if mime.startswith("image/"):
            return "image"
        return "document"
    return "other"


def forward_source(msg) -> tuple[int | None, int | None]:
    """(channel_id, post_id) when the message was forwarded from a channel; never a user id."""
    fwd = getattr(msg, "fwd_from", None)
    peer = getattr(fwd, "from_id", None)
    if fwd is None or type(peer).__name__ != "PeerChannel":
        return None, None
    return peer.channel_id, getattr(fwd, "channel_post", None)


def message_to_record(msg, channel: str, keep_text: bool = True) -> dict:
    """Static facts about a publication. These don't change after posting (except edits).

    keep_text=False drops the text of member chat messages in groups (counts are kept),
    so the database holds no member conversations.
    """
    text = getattr(msg, "message", None) or ""
    fwd_channel_id, fwd_channel_post = forward_source(msg)
    record = {
        "platform": "telegram",
        "channel": channel,
        "message_id": f"telegram:{channel}:{msg.id}",
        "telegram_id": msg.id,
        "published_at": _iso(msg.date),
        "edited_at": _iso(getattr(msg, "edit_date", None)),
        "message_text": text,
        "media_type": media_type(msg),
        "grouped_id": getattr(msg, "grouped_id", None),
        "pinned_flag": bool(getattr(msg, "pinned", False)),
        "is_forward": getattr(msg, "fwd_from", None) is not None,
        "fwd_channel_id": fwd_channel_id,
        "fwd_channel_post": fwd_channel_post,
        "character_count": len(text),
        "word_count": len(text.split()),
        "link_count": len(URL_RE.findall(text)),
        "poll_question": None,
        "poll_option_count": None,
        "poll_is_quiz": None,
    }
    if not keep_text and fwd_channel_id is None:
        record["message_text"] = None
        return record
    media = getattr(msg, "media", None)
    if record["media_type"] == "poll":
        poll = media.poll
        record["poll_question"] = _text_of(poll.question)
        record["poll_option_count"] = len(poll.answers or [])
        record["poll_is_quiz"] = bool(getattr(poll, "quiz", False))
    return record


def message_to_measurement(msg, channel: str, collected_at: datetime) -> dict:
    """Cumulative counters as observed at collected_at. Unavailable counters stay None, never 0."""
    reactions = getattr(msg, "reactions", None)
    by_reaction = {}
    if reactions is not None:
        for rc in reactions.results or []:
            by_reaction[_reaction_key(rc.reaction)] = rc.count

    replies = getattr(msg, "replies", None)
    poll_voters = None
    media = getattr(msg, "media", None)
    if media is not None and type(media).__name__ == "MessageMediaPoll":
        poll_voters = getattr(media.results, "total_voters", None)

    age_hours = (collected_at - msg.date).total_seconds() / 3600
    return {
        "message_id": f"telegram:{channel}:{msg.id}",
        "collected_at": _iso(collected_at),
        "age_hours": round(age_hours, 3),
        "views": getattr(msg, "views", None),
        "forwards": getattr(msg, "forwards", None),
        "reactions_total": sum(by_reaction.values()) if reactions is not None else None,
        "reactions_json": json.dumps(by_reaction, ensure_ascii=False) if reactions is not None else None,
        "comments": replies.replies if replies is not None and replies.comments else None,
        "poll_voters": poll_voters,
    }


def _reaction_key(reaction) -> str:
    name = type(reaction).__name__
    if name == "ReactionEmoji":
        return reaction.emoticon
    if name == "ReactionCustomEmoji":
        return f"custom:{reaction.document_id}"
    if name == "ReactionPaid":
        return "paid_star"
    return name


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()
