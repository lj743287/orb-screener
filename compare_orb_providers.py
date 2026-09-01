#!/usr/bin/env python3
"""Compare production Twelve Data ORB candidates with the free Alpaca run."""

import json
import os
from datetime import datetime, timezone

import pandas as pd

TD_FILE = os.environ.get("TD_CANDIDATES", "output/candidates.csv")
ALPACA_FILE = os.environ.get(
    "ALPACA_CANDIDATES",
    "comparison/alpaca/candidates.csv",
)
OUT_MD = os.environ.get(
    "COMPARISON_REPORT",
    "comparison/provider_comparison.md",
)
OUT_JSON = os.environ.get(
    "COMPARISON_JSON",
    "comparison/provider_comparison.json",
)


def read_candidates(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def symbols(df):
    if df.empty or "symbol" not in df.columns:
        return set()
    return set(df["symbol"].dropna().astype(str).str.upper())


def main():
    td = read_candidates(TD_FILE)
    ap = read_candidates(ALPACA_FILE)

    td_syms = symbols(td)
    ap_syms = symbols(ap)
    common = sorted(td_syms & ap_syms)
    td_only = sorted(td_syms - ap_syms)
    ap_only = sorted(ap_syms - td_syms)
    union = td_syms | ap_syms

    match_pct = 100.0 if not union else 100.0 * len(common) / len(union)

    metric_rows = []
    if common and not td.empty and not ap.empty:
        td_i = td.set_index(td["symbol"].astype(str).str.upper())
        ap_i = ap.set_index(ap["symbol"].astype(str).str.upper())
        for sym in common:
            row = {"symbol": sym}
            for col in ("price", "adr", "runup", "base_depth", "ext10", "dvolM"):
                if col not in td_i.columns or col not in ap_i.columns:
                    continue
                try:
                    a = float(td_i.loc[sym, col])
                    b = float(ap_i.loc[sym, col])
                    row[f"td_{col}"] = a
                    row[f"alpaca_{col}"] = b
                    row[f"diff_{col}"] = round(b - a, 4)
                except (TypeError, ValueError):
                    pass
            metric_rows.append(row)

    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "twelve_data_candidates": len(td_syms),
        "alpaca_candidates": len(ap_syms),
        "common_candidates": len(common),
        "union_candidates": len(union),
        "candidate_jaccard_match_pct": round(match_pct, 2),
        "twelve_data_only": td_only,
        "alpaca_only": ap_only,
        "common": common,
        "metrics": metric_rows,
    }

    os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    lines = [
        "# ORB provider comparison",
        "",
        f"Generated: {payload['generated_utc']}",
        "",
        "The **screening code is identical** in both runs. Only the daily-bar provider differs.",
        "",
        f"- Twelve Data candidates: **{len(td_syms)}**",
        f"- Alpaca candidates: **{len(ap_syms)}**",
        f"- Common candidates: **{len(common)}**",
        f"- Candidate-set match: **{match_pct:.1f}%** (intersection / union)",
        "",
        "## Twelve Data only",
        "",
        ", ".join(td_only) if td_only else "None",
        "",
        "## Alpaca only",
        "",
        ", ".join(ap_only) if ap_only else "None",
        "",
        "## Common candidates",
        "",
        ", ".join(common) if common else "None",
        "",
    ]

    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(
        f"Twelve Data {len(td_syms)} | Alpaca {len(ap_syms)} | "
        f"common {len(common)} | match {match_pct:.1f}%"
    )
    if td_only:
        print("TD only:", ", ".join(td_only))
    if ap_only:
        print("Alpaca only:", ", ".join(ap_only))


if __name__ == "__main__":
    main()
