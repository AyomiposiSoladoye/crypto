# Collecting Telegram channel data

This collector fills `data/community.db` with one row per channel post in `messages` and one row per observation in `measurements`, following the data dictionary in the requirements brief.

## 1. One-time setup

1. Go to https://my.telegram.org, sign in, open **API development tools** and create an app. Copy the `api_id` and `api_hash`.
2. Set up the environment:
   ```bash
   python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env    # fill in TG_API_ID, TG_API_HASH, TG_CHANNELS
   ```
3. Log in once: `python -m ingest.telegram.scrape login`. Telegram sends a code to your app. Enter it, plus your 2FA password if you have one.
   This creates `secrets/telegram.session`. **That file is a full login to your Telegram account.** It is git-ignored. Never commit it, upload it or share it. If it leaks, end the session under Telegram → Settings → Devices.

## 2. Backfill history

```bash
python -m ingest.telegram.scrape backfill                 # everything
python -m ingest.telegram.scrape backfill --since 2025-01-01
python -m ingest.telegram.scrape export                   # CSV for a quick look
```

## 3. Keep measuring new posts

```bash
python -m ingest.telegram.scrape snapshot --days 8
```

Run this **every hour** (cron, Windows Task Scheduler, or a small always-on machine). Each run:
- picks up new posts,
- records the channel's subscriber count,
- stores a fresh measurement for every post from the last 8 days.

The data layer then reads the 1h / 6h / 24h / 7d values from the measurement closest to each age.

## Things that matter for the analysis

| Issue | What the collector does | Why |
|---|---|---|
| Counters are **cumulative at scrape time** | Each measurement stores `collected_at` and `age_hours` | A backfilled 2024 post shows lifetime views, not 24h views. Old and new posts can't be compared on one window until snapshots exist. For history, compare **lifetime** values between posts older than ~7 days, when Telegram views have mostly levelled off. Check this in the audit. |
| Subscriber count at publish time | `channel_snapshots`, from today onward | Past subscriber counts aren't available from messages. If you're a channel **admin**, the channel statistics (followers graph) can fill this in, and we can add that pull later. |
| Albums | `grouped_id` is stored | One album post arrives as several messages. They are merged into one publication during feature building. |
| Edited posts | `edited_at`, and the text is refreshed on each run | The text you analyse may differ from what went out. Edited posts are flagged. |
| Missing ≠ zero | Counters the API doesn't return stay `NULL` | For example, `comments` is only set when a discussion group is linked. |
| Privacy | No user IDs, reactor lists, voter lists or `post_author` are stored | Only aggregate counts. This matches section 16 of the brief. |
| Rate limits | `flood_sleep_threshold=120`, `wait_time=1` | Telethon waits out short FloodWait errors. Use one account and don't run several collectors in parallel. |

## Scraping channels you don't run

Public channel posts and their counters can be read by any member account, and the same code works. Keep this to what the brief allows:
- use it for pattern discovery,
- keep to a small sample,
- store no member data,
- follow Telegram's API Terms of Service.

Only your own channel's history is used to train the prediction model.

## Groups vs channels

`TG_CHANNELS` can name a **broadcast channel**, where only admins post and every post has a view count, or a **group**, where members chat.
- Prediction needs the channel. Members' chat messages have no view counts.
- In a group, posts forwarded from channels are kept in full. Member chat text depends on `TG_MEMBER_TEXT`:
  - `redact` (default) keeps the text, with emails, phone and account numbers, wallet addresses and @usernames replaced by `<EMAIL>`, `<NUMBER>`, `<WALLET>` and `<USER>`. Names typed in plain words are not caught.
  - `drop` keeps no member text, only time, length and reactions.
- Re-running `backfill` re-applies the setting to everything already stored.

To find the official channel behind a community group:

```bash
python -m ingest.telegram.scrape sources
```

Then put that channel's username in `TG_CHANNELS` and run `backfill` again.
