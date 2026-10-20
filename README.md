# Crypto Social Media Intelligence

Community analytics and Fogg-based content analysis for a crypto exchange's Telegram community. This builds on the final-year project *Fogg Behavior-Based ML Model for Twitter Virality Prediction*.

Everything runs locally. Collected data, reports and models stay in `data/`, `reports/` and `models/`, which are not committed.

## Setup (Windows)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      (then fill it in, see docs/telegram-collection.md)
```

## Commands

| Step | Command | Output |
|---|---|---|
| Log in to Telegram once | `python -m ingest.telegram.scrape login` | `secrets/telegram.session` (keep private) |
| Collect history | `python -m ingest.telegram.scrape backfill` | `data/community.db` |
| Keep measuring (hourly) | `python -m ingest.telegram.scrape snapshot` | new measurement rows |
| Export to Excel | `python -m ingest.telegram.scrape export` | `data/telegram_messages.csv` |
| Community insights | `python -m analysis.report` | `reports/community_insights.md`, `data/clean/*.csv` |
| Train + evaluate model | `python -m model.train` | `reports/model_report.md`, `models/<version>/` |
| Check a draft | `python -m model.analyse "your post" --media image --time "2026-10-21 17:00"` | Fogg checklist (+ estimate if validated) |
| Run tests | `python -m pytest -q` | |

## How the model is judged

- Posts are split **by date**: the oldest 60% for training, the next 20% for choosing the model, and the newest 20% for the final test.
- The model counts as **validated** only if, on that newest period:
  - it beats simply predicting the historical rate, with a 95% bootstrap interval above zero, and
  - its probabilities are better calibrated (Brier score).
- If it isn't validated, `model.analyse` shows no estimate, only the Fogg checklist.
