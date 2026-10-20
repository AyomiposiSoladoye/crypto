"""Train and evaluate the above-baseline announcement model.

    python -m model.train                     # reads data/community.db
    python -m model.train --source data/telegram_messages.csv

Data is split by date: oldest 60% train, next 20% validation (model choice and
decision threshold), newest 20% test (final, untouched). The chosen model is only
marked "validated" if, on the test period, it beats the base rate with a 95%
bootstrap interval above zero and has a better Brier score.

Writes models/<version>/model.joblib + card.json and reports/model_report.md.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.insights import label_announcements
from analysis.load import load_messages, split

from .features import ALL_FEATURES, DIMENSION, FEATURE_GROUPS, FEATURE_VERSION, build_feature_table

SEED = 42
TARGET = "above_baseline"
CONTEXT_COLS = FEATURE_GROUPS["context"] + [f"media_{m}" for m in ("text", "image", "video", "poll", "link")]


def candidates() -> dict:
    impute = lambda: SimpleImputer(strategy="median")  # noqa: E731
    return {
        "logistic_regression": make_pipeline(impute(), StandardScaler(),
                                             LogisticRegression(C=0.3, max_iter=2000)),
        "random_forest": make_pipeline(impute(), RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, random_state=SEED, n_jobs=-1)),
        "gradient_boosting": make_pipeline(impute(), GradientBoostingClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=SEED)),
    }


def load_dataset(source: str) -> pd.DataFrame:
    """Labelled feature table. The top-25% cut-off is fixed from the training period only."""
    ann, _ = split(load_messages(source))
    data = build_feature_table(label_announcements(ann))
    data = data[data["relative_views"].notna()].reset_index(drop=True)
    train, _, _ = chrono_split(data)
    cutoff = float(train["relative_views"].quantile(0.75))
    data[TARGET] = (data["relative_views"] >= cutoff).astype(int)
    data.attrs["cutoff"] = cutoff
    return data


def chrono_split(data: pd.DataFrame, val=0.2, test=0.2):
    n = len(data)
    a, b = int(n * (1 - val - test)), int(n * (1 - test))
    return data.iloc[:a], data.iloc[a:b], data.iloc[b:]


def metrics(y, p, threshold) -> dict:
    pred = (p >= threshold).astype(int)
    return {
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"),
        "brier": brier_score_loss(y, p),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }


def best_threshold(y, p) -> float:
    grid = np.unique(np.round(p, 3))
    return float(max(grid, key=lambda t: f1_score(y, (p >= t).astype(int), zero_division=0)))


def bootstrap_lift(y, p, base_rate, n=1000) -> tuple[float, float]:
    """95% interval for PR-AUC minus the base rate (what a no-skill model scores)."""
    rng = np.random.default_rng(SEED)
    y, p = np.asarray(y), np.asarray(p)
    lifts = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        lifts.append(average_precision_score(y[i], p[i]) - y[i].mean())
    return float(np.percentile(lifts, 2.5)), float(np.percentile(lifts, 97.5))


def calibration_table(y, p, bins=5) -> pd.DataFrame:
    df = pd.DataFrame({"p": p, "y": y})
    df["bin"] = pd.qcut(df["p"], q=bins, duplicates="drop")
    g = df.groupby("bin", observed=True)
    return pd.DataFrame({"posts": g.size(), "avg_predicted": g["p"].mean().round(3),
                         "actual_rate": g["y"].mean().round(3)})


def run(source: str, out_dir: Path) -> dict:
    data = load_dataset(source)
    train, val, test = chrono_split(data)
    X = lambda d, cols=ALL_FEATURES: d[cols]  # noqa: E731

    # 1. Pick the model family on validation.
    val_scores = {}
    for name, model in candidates().items():
        model.fit(X(train), train[TARGET])
        val_scores[name] = average_precision_score(val[TARGET], model.predict_proba(X(val))[:, 1])
    chosen = max(val_scores, key=val_scores.get)

    # 2. Threshold on validation, then refit on train + validation and score the test period once.
    model = candidates()[chosen].fit(X(train), train[TARGET])
    threshold = best_threshold(val[TARGET], model.predict_proba(X(val))[:, 1])
    trainval = pd.concat([train, val])
    model = candidates()[chosen].fit(X(trainval), trainval[TARGET])
    p_test = model.predict_proba(X(test))[:, 1]

    # Baselines on the same test period.
    base_rate = trainval[TARGET].mean()
    # Same procedure for a model that only knows timing and format (no wording).
    ctx = candidates()["logistic_regression"].fit(X(train, CONTEXT_COLS), train[TARGET])
    ctx_threshold = best_threshold(val[TARGET], ctx.predict_proba(X(val, CONTEXT_COLS))[:, 1])
    ctx = candidates()["logistic_regression"].fit(X(trainval, CONTEXT_COLS), trainval[TARGET])
    p_context = ctx.predict_proba(X(test, CONTEXT_COLS))[:, 1]
    results = {
        "base_rate (always predict history)": metrics(test[TARGET], np.full(len(test), base_rate), 0.5),
        "timing + format only": metrics(test[TARGET], p_context, ctx_threshold),
        f"{chosen} (all Fogg features)": metrics(test[TARGET], p_test, threshold),
    }
    lift_lo, lift_hi = bootstrap_lift(test[TARGET], p_test, base_rate)
    validated = lift_lo > 0 and results[f"{chosen} (all Fogg features)"]["brier"] < results[
        "base_rate (always predict history)"]["brier"]

    cv = rolling_cv(data)

    imp = permutation_importance(model, X(test), test[TARGET], scoring="average_precision",
                                 n_repeats=20, random_state=SEED)
    importance = (pd.DataFrame({"feature": ALL_FEATURES, "importance": imp.importances_mean,
                                "std": imp.importances_std})
                  .assign(dimension=lambda d: d["feature"].map(DIMENSION))
                  .sort_values("importance", ascending=False))

    version = f"{datetime.now():%Y%m%d-%H%M}-{chosen}"
    folder = out_dir / "models" / version
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, folder / "model.joblib")
    last = data.iloc[-1]
    card = {
        "version": version,
        "created": datetime.now().isoformat(timespec="seconds"),
        "model": chosen,
        "feature_version": FEATURE_VERSION,
        "features": ALL_FEATURES,
        "label": "above_baseline = views / median of previous 30 announcements in the training period's top 25%",
        "label_threshold_relative_views": round(data.attrs["cutoff"], 3),
        "periods": {name: [str(d["first_forwarded_at"].min().date()), str(d["first_forwarded_at"].max().date()), len(d)]
                    for name, d in [("train", train), ("validation", val), ("test", test)]},
        "base_rate": round(float(base_rate), 3),
        "test_share_above_baseline": round(float(test[TARGET].mean()), 3),
        "decision_threshold": round(threshold, 3),
        "validation_pr_auc": {k: round(v, 3) for k, v in val_scores.items()},
        "test_metrics": {k: {m: round(v, 3) for m, v in r.items()} for k, r in results.items()},
        "test_pr_auc_lift_95ci": [round(lift_lo, 3), round(lift_hi, 3)],
        "validated": bool(validated),
        "rolling_cv_pr_auc_lift": cv.round(3).to_dict(orient="index"),
        "context_defaults": {
            "baseline_views": float(data["views"].tail(30).median()),
            "days_since_prev_post": float(data["days_since_prev_post"].tail(30).median()),
            "posts_prev_7d": int(data["posts_prev_7d"].tail(30).median()),
        },
    }
    card["reference"] = reference_stats(data)
    (folder / "card.json").write_text(json.dumps(card, indent=2))
    (out_dir / "models" / "LATEST").write_text(version)

    report = render(card, results, importance, calibration_table(test[TARGET], p_test), data, cv)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports" / "model_report.md").write_text(report, encoding="utf-8")
    return card


def reference_stats(data: pd.DataFrame) -> dict:
    """Typical feature values of past top posts vs the rest, for comparing a new draft."""
    top, rest = data[data[TARGET] == 1], data[data[TARGET] == 0]
    return {f: {"top_median": round(float(top[f].median()), 3), "rest_median": round(float(rest[f].median()), 3),
                "top_mean": round(float(top[f].mean()), 3), "rest_mean": round(float(rest[f].mean()), 3)}
            for f in ALL_FEATURES}


def rolling_cv(data: pd.DataFrame, folds: int = 5) -> pd.DataFrame:
    """PR-AUC minus the fold's share of top posts, across expanding-window time folds."""
    rows = {name: [] for name in [*candidates(), "timing + format only"]}
    for tr, te in TimeSeriesSplit(n_splits=folds).split(data):
        a, b = data.iloc[tr], data.iloc[te]
        if b[TARGET].nunique() < 2:
            continue
        prevalence = b[TARGET].mean()
        for name, model in candidates().items():
            p = model.fit(a[ALL_FEATURES], a[TARGET]).predict_proba(b[ALL_FEATURES])[:, 1]
            rows[name].append(average_precision_score(b[TARGET], p) - prevalence)
        m = candidates()["logistic_regression"].fit(a[CONTEXT_COLS], a[TARGET])
        rows["timing + format only"].append(
            average_precision_score(b[TARGET], m.predict_proba(b[CONTEXT_COLS])[:, 1]) - prevalence)
    return pd.DataFrame({k: {"mean_lift": np.mean(v), "min_lift": np.min(v), "max_lift": np.max(v),
                             "folds_above_zero": int(np.sum(np.array(v) > 0)), "folds": len(v)}
                         for k, v in rows.items()}).T


def render(card, results, importance, calib, data, cv) -> str:
    by_dim = importance.groupby("dimension")["importance"].apply(lambda s: s.clip(lower=0).sum())
    share = (by_dim / by_dim.sum() * 100).round(1) if by_dim.sum() > 0 else by_dim * 0
    verdict = (
        "**Validated.** On the newest, untouched period it ranks posts better than chance with a 95% "
        "interval above zero and has better-calibrated probabilities than the base rate. It can be "
        "shown in the app as an estimate, alongside its limits."
        if card["validated"] else
        "**Not validated.** On the newest, untouched period it does not reliably beat simply predicting "
        "the historical rate. Per the project's Gate 3, the app must not present its output as a "
        "prediction. Use the Fogg analysis and descriptive insights instead, and keep collecting data."
    )
    p = card["periods"]
    parts = [
        f"# Announcement model report — `{card['version']}`",
        f"_Features `{card['feature_version']}` · {len(data)} labelled announcements_",
        "## Verdict", verdict,
        f"Test PR-AUC lift over base rate: **{card['test_pr_auc_lift_95ci'][0]:+.3f} to "
        f"{card['test_pr_auc_lift_95ci'][1]:+.3f}** (95% bootstrap interval).",
        "## Setup",
        f"- Target: {card['label']} (relative views ≥ {card['label_threshold_relative_views']}).",
        f"- Train {p['train'][0]} → {p['train'][1]} ({p['train'][2]} posts); validation {p['validation'][0]} → "
        f"{p['validation'][1]} ({p['validation'][2]}); test {p['test'][0]} → {p['test'][1]} ({p['test'][2]}).",
        f"- Model chosen on validation PR-AUC: {card['validation_pr_auc']} → **{card['model']}**.",
        "## Test results (newest period)",
        "PR-AUC is the main score. A no-skill model scores the share of top posts in the test period. "
        "Brier: lower is better.",
        pd.DataFrame(results).T.round(3).to_markdown(),
        f"Share of top posts: {card['base_rate']:.0%} in train + validation, "
        f"{card['test_share_above_baseline']:.0%} in test. A gap means the period has drifted, and "
        "probabilities trained on the past will be off by about that much.",
        "## Stability check: 5 rolling time folds",
        "Each fold trains on everything before it and tests on the next block. Lift = PR-AUC minus "
        "the share of top posts in that block (0 = no better than chance).",
        cv.round(3).to_markdown(),
        "## What the model relies on (permutation importance on test)",
        "Share of positive importance by Fogg dimension:",
        share.rename("share_%").to_frame().to_markdown(),
        importance.head(15).round(4).set_index("feature").to_markdown(),
        "Importance shows what the model uses, not what causes engagement.",
        "## Calibration on test",
        "If the model is calibrated, `avg_predicted` ≈ `actual_rate` in each row.",
        calib.to_markdown(),
        "## Limits",
        "- Small dataset. Results can shift with a few posts.",
        "- Views are lifetime totals at scrape time, not a fixed 24 h window.",
        "- Only announcements forwarded into the community group are included.",
    ]
    return "\n\n".join(parts) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="data/community.db")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()
    card = run(args.source, Path(args.out_dir))
    print(f"Model {card['version']}: validated={card['validated']}")
    print(json.dumps(card["test_metrics"], indent=2))
    print("Report: reports/model_report.md")


if __name__ == "__main__":
    main()
