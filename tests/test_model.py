import numpy as np
import pandas as pd
import pytest

from model.features import ALL_FEATURES, text_features
from model.train import run

OUTCOMES = {"views", "forwards", "reactions_total", "poll_voters", "relative_views", "above_baseline"}


def test_no_outcome_columns_are_features():
    assert not OUTCOMES & set(ALL_FEATURES)


def test_fyp_bugs_fixed():
    assert text_features("Collect your badge today")["emotion_score"] == 0      # 'bad' inside 'badge'
    assert text_features("The httpd server is fine")["has_url"] == 0           # not a URL
    assert text_features("see https://quidax.com now")["has_url"] == 1
    assert text_features(float("nan"))["char_count"] == 0                      # captionless media


def test_poll_counts_as_vote_prompt():
    assert text_features("Which coin next?", "poll")["cta_vote"] == 1


def synthetic_export(path, signal: bool, n=420, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2024-01-01 09:00", periods=n, freq="20h", tz="UTC")
    rows = []
    for i, ts in enumerate(t):
        hype = rng.random() < 0.3
        text = ("Hey Quidaxians! 🎉 Big giveaway today, join now: https://t.me/x" if hype
                else "Market update for this week.")
        views = 1000 * (3 if (hype and signal) else 1) * rng.lognormal(0, 0.15)
        rows.append({"message_id": f"m{i}", "platform": "telegram", "channel": "grp", "telegram_id": i,
                     "published_at": ts.isoformat(), "message_text": text, "media_type": "text",
                     "is_forward": 1, "fwd_channel_id": 7, "fwd_channel_post": i, "character_count": len(text),
                     "link_count": 0, "views": round(views), "forwards": 1, "reactions_total": 2,
                     "poll_voters": None})
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.mark.parametrize("signal,expected", [(True, True), (False, False)])
def test_validation_gate(tmp_path, signal, expected):
    src = tmp_path / "export.csv"
    synthetic_export(src, signal)
    card = run(str(src), tmp_path)
    assert card["validated"] is expected
    assert (tmp_path / "models" / "LATEST").exists() and (tmp_path / "reports" / "model_report.md").exists()
    p = card["periods"]
    assert p["train"][1] <= p["validation"][0] <= p["validation"][1] <= p["test"][0]   # strictly by date
