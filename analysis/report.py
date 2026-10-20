"""Build clean datasets and a community insights report.

    python -m analysis.report                       # reads data/community.db
    python -m analysis.report --source data/telegram_messages.csv

Writes data/clean/announcements.csv, data/clean/chat_labelled.csv and
reports/community_insights.md. Everything runs locally.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from ingest.telegram.parse import redact

from . import insights as ins
from .classify import label_chat
from .load import load_messages, split

ISSUE_TOPICS = ["app_access", "deposit_withdrawal", "account_kyc"]


def build(source: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_messages(source)
    ann, chat = split(df)
    chat["message_text"] = chat["message_text"].fillna("").map(redact)  # in case of an older, unmasked export
    chat = label_chat(chat)
    member = chat[chat["kind"] == "member"]
    ann = ins.label_announcements(ann)
    ann["chat_lift_2h"] = ins.chat_response(ann, member)
    return ann, chat


def render(ann: pd.DataFrame, chat: pd.DataFrame, source: str) -> str:
    member = chat[chat["kind"] == "member"]
    kinds = chat["kind"].value_counts()
    first, last = chat["local_time"].min(), chat["local_time"].max()
    topics = member["topic"].value_counts(normalize=True).mul(100).round(1)
    a = ins.announcement_breakdowns(ann)
    md = lambda df: df.to_markdown()  # noqa: E731

    parts = [
        "# Community insights report",
        f"_Source: `{source}` · generated {datetime.now():%Y-%m-%d %H:%M} · times in Lagos time (WAT)_",
        "> Topic, question and staff labels come from keyword rules, and sentiment from VADER, which "
        "misreads Pidgin and sarcasm. Treat percentages as indicative and spot-check "
        "`data/clean/chat_labelled.csv` before quoting them.",
        "## 1. What's in the data",
        f"- Period: **{first:%d %b %Y} – {last:%d %b %Y}**",
        f"- Group messages: **{len(chat):,}**. Of these, **{kinds.get('member', 0):,}** are member chat; "
        f"{kinds.get('bot_command', 0):,} bot commands, {kinds.get('automated', 0):,} automated/bot replies, "
        f"{kinds.get('media_only', 0):,} media without text, {kinds.get('likely_staff', 0):,} likely staff replies.",
        f"- Announcements (posts forwarded from the group's main channel, de-duplicated): **{len(ann):,}**",
        "## 2. What members talk about",
        md(topics.rename("share_%").to_frame()),
        f"- {member['is_question'].mean():.0%} of member messages are questions; "
        f"{member['is_issue'].mean():.0%} are about app access, deposits/withdrawals or account verification.",
        "### Topic mix by quarter (% of member messages)",
        md(ins.topic_share_by_quarter(member)),
        "### Monthly trend",
        md(ins.monthly_trend(member)),
        "### Weeks with unusually many issue messages",
        "Likely incidents (outages, app migrations, bank delays). Standout words are those far more "
        "common that week than usual.",
        md(ins.weekly_issue_spikes(member)),
        "### Sentiment by topic",
        md(ins.sentiment_by_topic(member)),
        "### Recurring question phrases (content and FAQ ideas)",
    ]
    for topic in ISSUE_TOPICS + ["product_fees", "trading_market"]:
        parts += [f"**{topic}**", md(ins.recurring_question_phrases(member, topic).set_index("phrase"))]
    parts += [
        "### When members are active (messages by weekday × hour)",
        md(ins.activity_heatmap(member)),
        "## 3. Announcement performance",
        f"`relative_views` = views ÷ median views of the previous 30 announcements, so no future data "
        f"is used. `above_baseline` marks the top 25% (relative_views ≥ {ann.attrs['threshold']:.2f}). "
        "`chat_lift_2h` = member messages in the 2 h after a post minus the 2 h before.",
        "### By year", md(a["by_year"]),
        "### By format", md(a["by_format"]),
        "### By length", md(a["by_length"]),
        "### By posting hour (Lagos)", md(a["by_hour"]),
        "### Top 10 announcements relative to their baseline",
        md(ann.dropna(subset=["relative_views"]).nlargest(10, "relative_views")
           .assign(text=lambda d: d["message_text"].fillna("").str.replace("\n", " ").str[:90],
                   date=lambda d: d["first_forwarded_at"].dt.date)
           [["date", "media_type", "views", "relative_views", "chat_lift_2h", "text"]].set_index("date")),
        "## 4. Caveats",
        "- Announcement views are lifetime totals as of the scrape date, so recent posts have had less "
        "time to collect views. The trailing baseline partly offsets this.",
        "- Only announcements someone forwarded into the group are included, not every channel post.",
        "- Associations here are not causes. Test promising patterns before relying on them.",
    ]
    return "\n\n".join(parts) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="data/community.db")
    p.add_argument("--out-dir", default=".")
    args = p.parse_args()

    out = Path(args.out_dir)
    ann, chat = build(args.source)
    (out / "data/clean").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    ann.to_csv(out / "data/clean/announcements.csv", index=False, encoding="utf-8-sig")
    chat.drop(columns=["reactions_json"], errors="ignore").to_csv(
        out / "data/clean/chat_labelled.csv", index=False, encoding="utf-8-sig")
    report = out / "reports/community_insights.md"
    report.write_text(render(ann, chat, args.source), encoding="utf-8")
    print(f"Wrote {out / 'data/clean'}/announcements.csv, chat_labelled.csv and {report}")


if __name__ == "__main__":
    main()
