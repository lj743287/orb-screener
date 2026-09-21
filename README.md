# ORB Continuation Screener

Scans the NYSE/NASDAQ common-stock universe each evening (GitHub Actions) using
Alpaca daily bars, applies the validated entry parameters, and publishes a watchlist of
names *set up* for an opening-range-high break the next session.

Output: `output/candidates.csv` and a phone-friendly page at `docs/index.html`
(served via GitHub Pages).

## Setup (once)
1. Create a **public** repo (public = unlimited free Actions minutes) and add these files.
2. **Settings → Secrets and variables → Actions → New repository secret**
   - `APCA_API_KEY_ID` = your Alpaca API key ID
   - `APCA_API_SECRET_KEY` = your Alpaca secret key
   - *(optional)* add a **Variable** `MAX_SYMBOLS` (e.g. `300`) to cap the scan while testing / to stay inside your daily credit budget. Leave unset for the full universe.
3. **Settings → Pages → Source: Deploy from a branch → `main` / `docs`.**
   Your watchlist will live at `https://<you>.github.io/<repo>/`.
4. **Actions tab → enable workflows.** Click **Run workflow** to test now
   (set `MAX_SYMBOLS=50` first so the test is cheap).

## Schedule
Runs at `22:15 UTC, Monday–Friday`, safely after the US close in both daylight-saving
periods. It can also be started manually from the Actions tab.

## Credit budget
The Alpaca fetcher batches symbols and stays below the configured historical-data
request limit. Set `MAX_SYMBOLS` only when you deliberately want a smaller test run.

## Parameters
Edit the `P = dict(...)` block in `screener.py`:
`ADR_MAX, RUNUP_MIN, RUNUP_MAX, PRICE_MIN, BASE_MIN, RUNUP_LB, MA_TOL`.
