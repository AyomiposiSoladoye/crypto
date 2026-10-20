"""Analyse a draft announcement before publishing.

    python -m model.analyse "Hey Quidaxians! ..." --media image --time "2026-10-21 17:00"

Always shows the Fogg breakdown and how the draft compares with past top posts.
Shows a model estimate only if the latest model passed validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from analysis.load import LOCAL_TZ

from .features import ALL_FEATURES, context_features, text_features

# (feature, plain-language label, Fogg dimension) for the checklist.
CHECKS = [
    ("line_count", "Lines / paragraph breaks", "ability"),
    ("char_count", "Length (characters)", "ability"),
    ("avg_sentence_words", "Words per sentence", "ability"),
    ("jargon_density", "Crypto jargon per word", "ability"),
    ("has_url", "Has a link", "ability"),
    ("community_address", "Addresses the community (e.g. 'Quidaxians')", "motivation"),
    ("sentiment_compound", "Overall tone (-1 negative … +1 positive)", "motivation"),
    ("emoji_count", "Emojis", "motivation"),
    ("reward_count", "Reward words (giveaway, bonus, free…)", "motivation"),
    ("security_count", "Security words (scam, safe, alert…)", "motivation"),
    ("cta_count", "Calls to action", "trigger"),
    ("has_question", "Asks a question", "trigger"),
    ("has_deadline", "Mentions a time or deadline", "trigger"),
    ("mention_count", "@mentions", "trigger"),
]


def load_latest(models_dir: str | Path = "models") -> tuple[dict, object] | tuple[None, None]:
    models_dir = Path(models_dir)
    latest = models_dir / "LATEST"
    if not latest.exists():
        return None, None
    folder = models_dir / latest.read_text().strip()
    card = json.loads((folder / "card.json").read_text())
    return card, joblib.load(folder / "model.joblib")


def analyse_draft(text: str, media_type: str = "text", when: pd.Timestamp | None = None,
                  models_dir: str | Path = "models") -> dict:
    card, model = load_latest(models_dir)
    when = when or pd.Timestamp.now(tz=LOCAL_TZ)
    if when.tzinfo is None:
        when = when.tz_localize(LOCAL_TZ)
    feats = text_features(text, media_type)
    defaults = (card or {}).get("context_defaults", {"baseline_views": 1000, "days_since_prev_post": 1, "posts_prev_7d": 3})
    feats.update(context_features(when, defaults["days_since_prev_post"], defaults["posts_prev_7d"],
                                  defaults["baseline_views"]))

    comparison = []
    ref = (card or {}).get("reference", {})
    for f, label, dim in CHECKS:
        r = ref.get(f)
        comparison.append({
            "dimension": dim, "check": label, "draft": round(float(feats[f]), 3),
            "past_top_posts": r and r["top_mean" if f.startswith(("has_", "community")) else "top_median"],
            "other_posts": r and r["rest_mean" if f.startswith(("has_", "community")) else "rest_median"],
        })

    result = {"features": feats, "comparison": comparison, "model": None}
    if card is not None:
        result["model"] = {"version": card["version"], "validated": card["validated"]}
        if card["validated"]:
            p = float(model.predict_proba(pd.DataFrame([feats])[ALL_FEATURES])[:, 1][0])
            result["model"].update({"probability_above_baseline": round(p, 3), "base_rate": card["base_rate"]})
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text")
    ap.add_argument("--media", default="text", choices=["text", "image", "video", "poll", "link"])
    ap.add_argument("--time", help="planned publish time in Lagos time, e.g. '2026-10-21 17:00'")
    args = ap.parse_args()
    r = analyse_draft(args.text, args.media, pd.Timestamp(args.time) if args.time else None)

    print("\nFOGG CHECKLIST  (past top posts vs other posts are typical values; a guide, not a rule)\n")
    table = pd.DataFrame(r["comparison"]).set_index(["dimension", "check"])
    print(table.to_string())
    m = r["model"]
    print("\nPREDICTION")
    if m is None:
        print("  No trained model yet. Run: python -m model.train")
    elif not m["validated"]:
        print(f"  Model {m['version']} did not pass validation, so no estimate is shown.")
        print("  See reports/model_report.md.")
    else:
        print(f"  {m['probability_above_baseline']:.0%} chance of beating the recent baseline "
              f"(typical post: {m['base_rate']:.0%}). Model {m['version']}.")


if __name__ == "__main__":
    main()
