# ORB Continuation Screener

Scans the NYSE/NASDAQ common-stock universe each evening (GitHub Actions) using
Twelve Data, applies the validated entry parameters, and publishes a watchlist of
names *set up* for an opening-range-high break the next session.

Output: `output/candidates.csv` and a phone-friendly page at `docs/index.html`
(served via GitHub Pages).

## Setup (once)
1. Create a **public** repo (public = unlimited free Actions minutes) and add these files.
2. **Settings → Secrets and variables → Actions → New repository secret**
   - `TWELVE_DATA_KEY` = your Twelve Data API key
   - *(optional)* add a **Variable** `MAX_SYMBOLS` (e.g. `300`) to cap the scan while testing / to stay inside your daily credit budget. Leave unset for the full universe.
3. **Settings → Pages → Source: Deploy from a branch → `main` / `docs`.**
   Your watchlist will live at `https://<you>.github.io/<repo>/`.
4. **Actions tab → enable workflows.** Click **Run workflow** to test now
   (set `MAX_SYMBOLS=50` first so the test is cheap).

## Schedule
Runs `02:00 UTC, Tue–Sat` (≈ evening ET, after the close). Cron is always UTC, so
the ET time shifts with daylight saving — adjust `cron` in `.github/workflows/screen.yml`
if you care about the exact hour.

## Credit budget
The full universe is ~8,000 symbols ≈ 8,000 Twelve Data credits per run. If that
exceeds your plan's **daily** allowance, set `MAX_SYMBOLS`, or pre-filter the
universe in `load_universe()`. `THROTTLE_SEC` spaces calls; the fetcher also
auto-waits and retries when it hits the per-minute limit.

## Parameters
Edit the `P = dict(...)` block in `screener.py`:
`ADR_MAX, RUNUP_MIN, RUNUP_MAX, PRICE_MIN, BASE_MIN, RUNUP_LB, MA_TOL`.
