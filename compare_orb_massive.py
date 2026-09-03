#!/usr/bin/env python3
"""Compare production Twelve Data ORB candidates with the Massive free run."""

import json
import os
from datetime import datetime, timezone

import pandas as pd

TD_FILE = os.environ.get("TD_CANDIDATES", "output/candidates.csv")
MASSIVE_FILE = os.environ.get(
    "MASSIVE_CANDIDATES",
    "comparison/massive/candidates.csv",
)
OUT_MD = os.environ.get(
    "COMPARISON_REPORT",
    "comparison/massive/provider_comparison.md",
)
OUT_JSON = os.environ.get(
    "COMPARISON_JSON",
    "comparison/massive/provider_comparison.json",
)

METRICS = ("price", "adr", "runup", "base_depth", "ext10", "dvolM")


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
    ms = read_candidates(MASSIVE_FILE)

    td_syms = symbols(td)
    ms_syms = symbols(ms)
    common = sorted(td_syms & ms_syms)
    td_only = sorted(td_syms - ms_syms)
    ms_only = sorted(ms_syms - td_syms)
    union = td_syms | ms_syms

    jaccard = 100.0 if not union else 100.0 * len(common) / len(union)
    td_recall = 100.0 if not td_syms else 100.0 * len(common) / len(td_syms)
    massive_recall = 100.0 if not ms_syms else 100.0 * len(common) / len(ms_syms)

    metric_rows = []
    if common and not td.empty and not ms.empty:
        td_i = td.set_index(td["symbol"].astype(str).str.upper())
        ms_i = ms.set_index(ms["symbol"].astype(str).str.upper())
        for sym in common:
            row = {"symbol": sym}
            for col in METRICS:
                if col not in td_i.columns or col not in ms_i.columns:
                    continue
                try:
                    a = float(td_i.loc[sym, col])
                    b = float(ms_i.loc[sym, col])
                    row[f"td_{col}"] = a
                    row[f"massive_{col}"] = b
                    row[f"diff_{col}"] = round(b - a, 4)
                except (TypeError, ValueError):
                    pass
            metric_rows.append(row)

    diff_summary = {}
    if metric_rows:
        md = pd.DataFrame(metric_rows)
        for col in METRICS:
            diff_col = f"diff_{col}"
            if diff_col not in md.columns:
                continue
            vals = pd.to_numeric(md[diff_col], errors="coerce").dropna().abs()
            if vals.empty:
                continue
            diff_summary[col] = {
                "mean_abs_diff": round(float(vals.mean()), 4),
                "median_abs_diff": round(float(vals.median()), 4),
                "max_abs_diff": round(float(vals.max()), 4),
            }

    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "twelve_data_candidates": len(td_syms),
        "massive_candidates": len(ms_syms),
        "common_candidates": len(common),
        "union_candidates": len(union),
        "candidate_jaccard_match_pct": round(jaccard, 2),
        "twelve_candidate_recall_pct": round(td_recall, 2),
        "massive_candidate_recall_pct": round(massive_recall, 2),
        "twelve_data_only": td_only,
        "massive_only": ms_only,
        "common": common,
        "metric_abs_diff_summary": diff_summary,
        "metrics": metric_rows,
    }

    os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    lines = [
        "# ORB provider comparison: Twelve Data vs Massive",
        "",
        f"Generated: {payload['generated_utc']}",
        "",
        "The **screening code is identical** in both runs. Only the daily-bar provider differs.",
        "",
        f"- Twelve Data candidates: **{len(td_syms)}**",
        f"- Massive candidates: **{len(ms_syms)}**",
        f"- Common candidates: **{len(common)}**",
        f"- Candidate-set match: **{jaccard:.1f}%** (intersection / union)",
        f"- Twelve candidate retention: **{td_recall:.1f}%**",
        f"- Massive candidate retention: **{massive_recall:.1f}%**",
        "",
        "## Common-candidate metric differences",
        "",
    ]

    if diff_summary:
        lines += [
            "| Metric | Mean absolute diff | Median absolute diff | Max absolute diff |",
            "|---|---:|---:|---:|",
        ]
        for col in METRICS:
            s = diff_summary.get(col)
            if not s:
                continue
            lines.append(
                f"| {col} | {s['mean_abs_diff']:.4f} | "
                f"{s['median_abs_diff']:.4f} | {s['max_abs_diff']:.4f} |"
            )
    else:
        lines.append("No common candidates available for metric comparison.")

    lines += [
        "",
        "## Twelve Data only",
        "",
        ", ".join(td_only) if td_only else "None",
        "",
        "## Massive only",
        "",
        ", ".join(ms_only) if ms_only else "None",
        "",
        "## Common candidates",
        "",
        ", ".join(common) if common else "None",
        "",
    ]

    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(
        f"Twelve Data {len(td_syms)} | Massive {len(ms_syms)} | "
        f"common {len(common)} | match {jaccard:.1f}% | "
        f"TD retention {td_recall:.1f}%"
    )
    if td_only:
        print("TD only:", ", ".join(td_only))
    if ms_only:
        print("Massive only:", ", ".join(ms_only))


if __name__ == "__main__":
    main()
