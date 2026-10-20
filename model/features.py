"""Fogg Behavior Model features for community announcements.

Ported from the final-year project's FoggFeatureExtractor, with these changes:
- emotion and CTA words match whole words only (the original matched substrings: "bad" in "badge"),
- URL detection uses a real pattern instead of any "http"/"www" substring,
- no engagement counts (likes/retweets) as inputs, since those are the outcome,
- length thresholds suited to Telegram posts rather than tweets,
- crypto/community-specific motivation, ability and trigger features,
- pre-publication context (timing, cadence, recent baseline).

Only information available before a post is published is used.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

FEATURE_VERSION = "fogg-v2.0"


def _rx(*words: str) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(words) + r")\b", re.IGNORECASE)


EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E0-\U0001F1FF⭐⭕⏰-⏺]"
)
URL = re.compile(r"https?://\S+|www\.\S+|\bt\.me/\S+", re.IGNORECASE)
HASHTAG = re.compile(r"#\w+")
MENTION = re.compile(r"@\w+")
NUMBER = re.compile(r"\d[\d,.]*%?")
SENTENCE_END = re.compile(r"[.!?]+(?:\s|$)")

POSITIVE = _rx("love", "amazing", "awesome", "great", "excellent", "exciting", "excited", "happy",
               "congrat\\w*", "celebrat\\w*", "win", "winner\\w*", "best", "thrilled")
NEGATIVE = _rx("hate", "terrible", "awful", "bad", "horrible", "sad", "angry", "loss", "losses",
               "fail\\w*", "worst")
URGENCY = _rx("now", "today", "tonight", "hurry", "last chance", "limited", "ends?", "ending",
              "deadline", "don'?t miss", "only", "before", "soon")
REWARD = _rx("giveaway\\w*", "win", "winners?", "rewards?", "bonus\\w*", "free", "prizes?",
             "airdrop\\w*", "cashback", "fee-free", "zero fees?", "promo\\w*", "earn")
SECURITY = _rx("scam\\w*", "fake", "security", "secure", "protect\\w*", "safe\\w*", "warning",
               "alert", "beware", "2fa", "phishing", "impersonat\\w*", "never ask")
EDUCATION = _rx("learn\\w*", "class", "academy", "how to", "guide", "explain\\w*", "tips?",
                "lessons?", "masterclass", "whitepaper", "did you know", "what is", "quiz")
MARKET = _rx("btc", "bitcoin", "eth", "ethereum", "price", "market", "etf", "halving", "rally",
             "bull\\w*", "bear\\w*", "listing", "listed", "delist\\w*", "upgrade", "usdt", "sol")
SERVICE = _rx("maintenance", "downtime", "delay\\w*", "resolved", "restored", "issue\\w*",
              "update", "new app", "migration", "outage", "inconvenience")
COMMUNITY = _rx("quidaxians?", "community", "fam", "family", "everyone", "guys", "friends")
SECOND_PERSON = _rx("you", "your", "you'?re", "yours")

# Words a newcomer may not know. Density = jargon words / words.
JARGON = _rx("blockchain", "defi", "nft\\w*", "staking", "stake", "liquidity", "altcoins?", "alts",
             "memecoins?", "tokenomics", "dex", "cex", "kyc", "p2p", "spot", "futures", "leverage",
             "gas", "network fee", "mainnet", "testnet", "layer ?2", "l2", "hodl", "dyor", "fud",
             "fomo", "ath", "apy", "apr", "whitepaper", "wallet address", "seed phrase", "private key",
             "order ?book", "limit order", "market order", "stablecoins?", "halving", "etf", "airdrop")

CTA_VOTE = _rx("vote", "poll", "pick", "choose")
CTA_REPLY = _rx("reply", "comment", "tell us", "drop (?:a|your)", "let us know", "share your", "answer")
CTA_LINK = _rx("click", "tap", "visit", "read more", "link", "download", "register", "sign up",
               "join", "watch", "check out", "learn more", "update your app")
CTA_SHARE = _rx("share", "invite", "refer\\w*", "tag a friend", "forward")
DEADLINE = re.compile(
    r"\b(?:today|tonight|tomorrow|deadline|ends?|until|by \d|\d{1,2}(?::\d{2})?\s?(?:am|pm)|"
    r"(?:mon|tues|wednes|thurs|fri|satur|sun)day)\b", re.IGNORECASE)

MEDIA_TYPES = ["text", "image", "video", "poll", "link"]

FEATURE_GROUPS = {
    "motivation": [
        "sentiment_compound", "sentiment_positive", "sentiment_negative", "emotion_score",
        "emoji_count", "has_emoji", "exclamation_count", "capital_ratio",
        "urgency_count", "reward_count", "security_count", "education_count", "market_count",
        "service_count", "community_address", "second_person_ratio",
    ],
    "ability": [
        "char_count", "word_count", "avg_word_length", "line_count", "avg_sentence_words",
        "jargon_density", "number_count", "url_count", "has_url", "is_short", "is_long",
        *[f"media_{m}" for m in MEDIA_TYPES],
    ],
    "trigger": [
        "question_count", "has_question", "hashtag_count", "mention_count", "cta_count",
        "cta_vote", "cta_reply", "cta_link", "cta_share", "has_deadline",
    ],
    "context": [
        "local_hour", "is_evening", "is_weekend", "days_since_prev_post", "posts_prev_7d",
        "log_baseline_views",
    ],
}
ALL_FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]
DIMENSION = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}

_vader = SentimentIntensityAnalyzer()


def text_features(text: str, media_type: str = "text") -> dict:
    """Motivation, ability and trigger features for one message."""
    text = text if isinstance(text, str) else ""  # media posts may have no caption
    words = text.split()
    n_words = max(len(words), 1)
    s = _vader.polarity_scores(text)
    sentences = max(len(SENTENCE_END.findall(text)), 1)
    emojis = EMOJI.findall(text)
    letters = [c for c in text if c.isalpha()]
    ctas = {name: len(rx.findall(text)) for name, rx in
            [("cta_vote", CTA_VOTE), ("cta_reply", CTA_REPLY), ("cta_link", CTA_LINK), ("cta_share", CTA_SHARE)]}
    f = {
        # Motivation
        "sentiment_compound": s["compound"],
        "sentiment_positive": s["pos"],
        "sentiment_negative": s["neg"],
        "emotion_score": len(POSITIVE.findall(text)) - len(NEGATIVE.findall(text)),
        "emoji_count": len(emojis),
        "has_emoji": int(bool(emojis)),
        "exclamation_count": text.count("!"),
        "capital_ratio": sum(c.isupper() for c in letters) / max(len(letters), 1),
        "urgency_count": len(URGENCY.findall(text)),
        "reward_count": len(REWARD.findall(text)),
        "security_count": len(SECURITY.findall(text)),
        "education_count": len(EDUCATION.findall(text)),
        "market_count": len(MARKET.findall(text)),
        "service_count": len(SERVICE.findall(text)),
        "community_address": int(bool(COMMUNITY.search(text))),
        "second_person_ratio": len(SECOND_PERSON.findall(text)) / n_words,
        # Ability
        "char_count": len(text),
        "word_count": len(words),
        "avg_word_length": float(np.mean([len(w) for w in words])) if words else 0.0,
        "line_count": text.count("\n") + 1 if text else 0,
        "avg_sentence_words": len(words) / sentences,
        "jargon_density": len(JARGON.findall(text)) / n_words,
        "number_count": len(NUMBER.findall(text)),
        "url_count": len(URL.findall(text)),
        "has_url": int(bool(URL.search(text))),
        "is_short": int(len(text) < 150),
        "is_long": int(len(text) > 600),
        **{f"media_{m}": int(media_type == m) for m in MEDIA_TYPES},
        # Trigger
        "question_count": text.count("?"),
        "has_question": int("?" in text),
        "hashtag_count": len(HASHTAG.findall(text)),
        "mention_count": len(MENTION.findall(text)),
        "cta_count": sum(ctas.values()),
        **{k: int(v > 0) for k, v in ctas.items()},
        "has_deadline": int(bool(DEADLINE.search(text))),
    }
    if media_type == "poll":
        f["cta_vote"] = 1
    return f


def context_features(local_time: pd.Timestamp, days_since_prev_post: float,
                     posts_prev_7d: int, baseline_views: float) -> dict:
    return {
        "local_hour": local_time.hour,
        "is_evening": int(17 <= local_time.hour <= 21),
        "is_weekend": int(local_time.dayofweek >= 5),
        "days_since_prev_post": days_since_prev_post,
        "posts_prev_7d": posts_prev_7d,
        "log_baseline_views": math.log1p(baseline_views),
    }


def build_feature_table(ann: pd.DataFrame) -> pd.DataFrame:
    """Features for every labelled announcement (output of analysis.insights.label_announcements)."""
    ann = ann.sort_values("first_forwarded_at").reset_index(drop=True)
    t = ann["first_forwarded_at"]
    gap_days = t.diff().dt.total_seconds() / 86400
    prev_7d = [int(((t < x) & (t >= x - pd.Timedelta(days=7))).sum()) for x in t]
    rows = []
    for i, r in ann.iterrows():
        f = text_features(r["message_text"], r["media_type"])
        f.update(context_features(r["local_time"], gap_days.iloc[i] if i else np.nan,
                                  prev_7d[i], r["baseline_views"] if pd.notna(r["baseline_views"]) else np.nan))
        rows.append(f)
    feats = pd.DataFrame(rows)[ALL_FEATURES]
    feats["days_since_prev_post"] = feats["days_since_prev_post"].clip(upper=30)
    return pd.concat([ann, feats], axis=1)
