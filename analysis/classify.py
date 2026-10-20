"""Rule-based labelling of community chat. Runs locally; no text leaves the machine.

Every rule here is a keyword heuristic. Labels are a first pass to be spot-checked,
not ground truth.
"""
from __future__ import annotations

import re

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def _rx(*words: str) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(words) + r")\b", re.IGNORECASE)


# Messages produced by bots or posted from templates, not by members.
AUTOMATED = [
    re.compile(r"price: \$?[\d.]+.*(?:change|volume)", re.IGNORECASE | re.DOTALL),   # price-bot ticker
    re.compile(r"click on one of these coins", re.IGNORECASE),
    re.compile(r"^dear user", re.IGNORECASE),
    re.compile(r"admins will never message you first", re.IGNORECASE),
    re.compile(r"^reported .* to admins", re.IGNORECASE),
]

# Phrases typical of staff/moderator replies. Heuristic: members rarely write like this.
STAFF = _rx(
    r"we apologi[sz]e", r"please send (?:me|us) a dm", r"kindly reach out to (?:the )?support",
    r"rest assured", r"your funds are safe with us", r"we(?:'re| are) (?:working on|looking into)",
    r"our team is", r"thank you for your patience", r"kindly send (?:me|us) a dm",
)

# Primary topic = first match in this order.
TOPICS: list[tuple[str, re.Pattern]] = [
    ("scam_security", _rx(r"scam\w*", r"fake", r"hack\w*", r"phish\w*", r"impersonat\w*",
                          r"fraud\w*", r"theft", r"stole\w*", r"thie[fv]\w*")),
    ("app_access", _rx(r"log ?in", r"login", r"sign ?in", r"can'?t access", r"not opening",
                       r"not (?:working|loading)", r"website", r"vpn", r"new app", r"2fa", r"otp",
                       r"password", r"error", r"crash\w*", r"glitch\w*", r"down", r"update the app")),
    ("deposit_withdrawal", _rx(r"deposit\w*", r"withdr[ae]w\w*", r"transfer\w*", r"credited",
                               r"pending", r"refund\w*", r"missing", r"locked", r"on hold", r"not received",
                               r"bank", r"payout", r"funds?", r"my money")),
    ("account_kyc", _rx(r"kyc", r"verif\w*", r"re-?verif\w*", r"bvn", r"nin", r"selfie",
                        r"id card", r"restrict\w*", r"blocked", r"suspend\w*", r"limit\w*")),
    ("product_fees", _rx(r"fees?", r"swap\w*", r"p2p", r"stak\w*", r"earn", r"listing", r"list(?:ed)?",
                         r"delist\w*", r"road ?map", r"feature\w*", r"order ?book", r"rates?", r"spread")),
    ("promotions", _rx(r"giveaway\w*", r"airdrop\w*", r"rewards?", r"bonus\w*", r"quiz", r"ama",
                       r"competition", r"referr\w*", r"win(?:ner|ners|ning)?")),
    ("trading_market", _rx(r"price", r"pump\w*", r"dump\w*", r"bull\w*", r"bear\w*", r"dip",
                           r"ath", r"moon\w*", r"alts?", r"altcoins?", r"chart\w*", r"rally",
                           r"market", r"btc", r"bitcoin", r"eth", r"sol", r"usdt", r"trump", r"qdx",
                           r"\d+k", r"buy\w*", r"sell\w*", r"hold\w*")),
]
ISSUE_TOPICS = {"app_access", "deposit_withdrawal", "account_kyc"}

GREETING = re.compile(
    r"^(?:hello|hi|hey|gm|good (?:morning|afternoon|evening|night)|ok(?:ay)?|yes|no|thanks?|"
    r"thank you|lol|bro|boss|sir|alright|noted|wow|nice|amen|same|true)\W*$",
    re.IGNORECASE,
)
QUESTION_START = re.compile(
    r"^(?:how|what|why|when|where|who|which|can|could|is|are|does|do|did|will|should|pls how|please how)\b",
    re.IGNORECASE,
)

_vader = SentimentIntensityAnalyzer()


def primary_topic(text: str) -> str:
    if GREETING.match(text.strip()):
        return "greeting_social"
    for name, rx in TOPICS:
        if rx.search(text):
            return name
    return "other"


def label_chat(chat: pd.DataFrame) -> pd.DataFrame:
    """Adds kind, topic, is_question, is_issue and sentiment columns to member chat rows."""
    out = chat.copy()
    text = out["message_text"].fillna("")
    stripped = text.str.strip()

    repeated = stripped.str.lower().map(stripped.str.lower().value_counts())
    automated = text.map(lambda t: any(rx.search(t) for rx in AUTOMATED)) | (
        (repeated >= 20) & (stripped.str.len() > 40)  # canned message posted many times
    )
    out["kind"] = "member"
    out.loc[text.map(lambda t: bool(STAFF.search(t))), "kind"] = "likely_staff"
    out.loc[automated, "kind"] = "automated"
    out.loc[stripped.str.startswith("/"), "kind"] = "bot_command"
    out.loc[stripped.eq(""), "kind"] = "media_only"

    out["topic"] = text.map(primary_topic)
    out.loc[out["kind"] != "member", "topic"] = None
    out["is_question"] = (out["kind"] == "member") & (
        stripped.str.contains(r"\?", regex=True) | stripped.str.match(QUESTION_START)
    )
    out["is_issue"] = out["topic"].isin(ISSUE_TOPICS)
    out["sentiment"] = text.map(lambda t: _vader.polarity_scores(t)["compound"] if t else None)
    out.loc[out["kind"] != "member", "sentiment"] = None
    return out
