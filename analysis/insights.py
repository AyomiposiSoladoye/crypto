"""Community and announcement insights computed from labelled data."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

STOPWORDS = set("""
a an the and or but if so to of in on at for with from by is are was were be been am i im i'm
you your u ur we our us they them their it its it's this that these those my me mine he she his
her do does did doing have has had not no yes can can't cant could would should will just pls
please abeg what why how when where who which there here dey na una ooo oo o now still again
also too all any some one get got make made go going want need know like see time day today
quidax user good morning afternoon evening https http www com play
""".split())


def topic_share_by_quarter(member: pd.DataFrame) -> pd.DataFrame:
    q = member["local_time"].dt.tz_localize(None).dt.to_period("Q").astype(str)
    return (pd.crosstab(q, member["topic"], normalize="index") * 100).round(1)


def weekly_issue_spikes(member: pd.DataFrame, top: int = 8) -> pd.DataFrame:
    """Weeks where issue messages ran far above normal, with the words that stood out."""
    week = member["local_time"].dt.tz_localize(None).dt.to_period("W").dt.start_time
    issues = member[member["is_issue"]]
    counts = issues.groupby(week[member["is_issue"]]).size()
    median = counts.median()
    mad = (counts - median).abs().median() or 1
    spikes = counts[(counts - median) / mad > 3].sort_values(ascending=False).head(top)

    names = _proper_nouns(issues["message_text"])
    overall = Counter(w for t in issues["message_text"] for w in _words(t))
    total = sum(overall.values())
    rows = []
    for start, n in spikes.items():
        texts = issues.loc[week[member["is_issue"]] == start, "message_text"]
        wc = Counter(w for t in texts for w in _words(t))
        wt = sum(wc.values()) or 1
        lift = {w: (c / wt) / (overall[w] / total) for w, c in wc.items() if c >= 5 and w not in names}
        rows.append({
            "week_of": start.date(), "issue_messages": int(n), "normal_week": int(median),
            "standout_words": ", ".join(sorted(lift, key=lift.get, reverse=True)[:6]),
        })
    return pd.DataFrame(rows)


def sentiment_by_topic(member: pd.DataFrame) -> pd.DataFrame:
    g = member.groupby("topic")["sentiment"]
    out = pd.DataFrame({"messages": g.size(), "avg_sentiment": g.mean().round(3),
                        "share_negative_%": g.apply(lambda s: (s < -0.05).mean() * 100).round(1)})
    return out.sort_values("messages", ascending=False)


def monthly_trend(member: pd.DataFrame) -> pd.DataFrame:
    month = member["local_time"].dt.tz_localize(None).dt.to_period("M").astype(str)
    g = member.groupby(month)
    return pd.DataFrame({
        "member_messages": g.size(),
        "issue_%": (g["is_issue"].mean() * 100).round(1),
        "question_%": (g["is_question"].mean() * 100).round(1),
        "avg_sentiment": g["sentiment"].mean().round(3),
    })


def activity_heatmap(member: pd.DataFrame) -> pd.DataFrame:
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    t = member["local_time"]
    table = pd.crosstab(t.dt.dayofweek.map(dict(enumerate(days))), t.dt.hour)
    return table.reindex(days)


def recurring_question_phrases(member: pd.DataFrame, topic: str, top: int = 10) -> pd.DataFrame:
    qs = member[(member["topic"] == topic) & member["is_question"]]["message_text"]
    grams = Counter()
    for t in qs:
        w = [x for x in _words(t) if x not in STOPWORDS]
        grams.update(" ".join(w[i:i + 2]) for i in range(len(w) - 1))
    return pd.DataFrame(grams.most_common(top), columns=["phrase", "questions"])


def label_announcements(ann: pd.DataFrame, window: int = 30, top_share: float = 0.25) -> pd.DataFrame:
    """Views relative to the median of the previous `window` announcements (no future data),
    and a top-quartile 'above_baseline' label for later modelling."""
    out = ann.sort_values("first_forwarded_at").reset_index(drop=True)
    baseline = out["views"].shift(1).rolling(window, min_periods=10).median()
    out["baseline_views"] = baseline
    out["relative_views"] = (out["views"] / baseline).round(3)
    threshold = out["relative_views"].quantile(1 - top_share)
    out["above_baseline"] = (out["relative_views"] >= threshold).astype("Int64")
    out.loc[out["relative_views"].isna(), "above_baseline"] = pd.NA
    out["reactions_per_1k_views"] = (out["reactions_total"] / out["views"] * 1000).round(2)
    out.attrs["threshold"] = threshold
    return out


def chat_response(ann: pd.DataFrame, member: pd.DataFrame, hours: int = 2) -> pd.Series:
    """Member messages in the `hours` after each announcement minus the `hours` before."""
    ts = np.sort(member["published_at"].values)
    at = ann["first_forwarded_at"].values
    span = np.timedelta64(hours, "h")
    after = np.searchsorted(ts, at + span) - np.searchsorted(ts, at)
    before = np.searchsorted(ts, at) - np.searchsorted(ts, at - span)
    return pd.Series(after - before, index=ann.index)


def announcement_breakdowns(ann: pd.DataFrame) -> dict[str, pd.DataFrame]:
    valid = ann[ann["relative_views"].notna()]

    def by(col):
        g = valid.groupby(col)
        return pd.DataFrame({"posts": g.size(), "median_relative_views": g["relative_views"].median().round(2),
                             "top_quartile_%": (g["above_baseline"].mean() * 100).round(1),
                             "median_chat_lift_2h": g["chat_lift_2h"].median()}).sort_values("posts", ascending=False)

    year = ann.assign(year=ann["first_forwarded_at"].dt.year).groupby("year")
    return {
        "by_year": pd.DataFrame({"posts": year.size(), "median_views": year["views"].median(),
                                 "median_reactions_per_1k_views": year["reactions_per_1k_views"].median()}),
        "by_format": by("media_type"),
        "by_hour": by(valid["local_time"].dt.hour.rename("local_hour")),
        "by_length": by(pd.cut(valid["character_count"], [-1, 150, 400, 800, 10_000],
                               labels=["<150", "150-400", "400-800", "800+"]).rename("characters")),
    }


def _proper_nouns(texts: pd.Series) -> set[str]:
    """Words usually written capitalised mid-sentence: mostly people's names, so never surfaced."""
    caps, seen = Counter(), Counter()
    for t in texts:
        for m in re.finditer(r"(?<=[a-z,] )([A-Za-z][a-z']{2,})", t or ""):
            w = m.group(1)
            seen[w.lower()] += 1
            caps[w.lower()] += w[0].isupper()
    return {w for w, n in seen.items() if n >= 3 and caps[w] / n >= 0.5}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z']{2,}", (text or "").lower())
