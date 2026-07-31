#!/usr/bin/env python3
"""
build_scans.py — assembles the combined scan dashboard.

Reads the four shared-format files written by the overnight scans:

    data/orb.json           (screener.py)
    data/daily.json         (daily_scanner.py)
    data/burst.json         (burst_scan.py)
    data/anticipation.json  (anticipation_scan.py)

and writes, into docs/ (the folder GitHub Pages serves):

    scans.html              five tabs: ORB, Daily, Burst, Combined, Anticipation
    scans_orb.txt           TradingView import lists, EXCHANGE:SYMBOL
    scans_daily.txt         comma-separated
    scans_burst.txt
    scans_combined.txt      ORB + Daily + Burst, duplicates removed
    scans_anticipation.txt

This script fetches no market data and needs no API key. It only reads files
the scans already produced, so it is safe to run at any time -- rerunning it
just rebuilds the page from the latest results.

Standard library only.
"""

import html
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

DATA_DIR = "data"
OUT_DIR = "docs"

# Tab order on the page. The combined tab is inserted between Burst and
# Anticipation by build_page().
SCANS = [
    ("orb", "ORB"),
    ("daily", "Daily"),
    ("burst", "Burst"),
    ("anticipation", "Anticipation"),
    ("burst_check", "Second burst"),
]

# The intraday tab. Everything else reflects last night's close and sits
# still all day; this one is refreshed during the session.
INTRADAY = {"burst_check"}

# Which scans feed the combined watchlist. Anticipation is deliberately left
# out -- those are coiled setups waiting on a pivot break, a different list
# from "these are moving now".
COMBINED_SOURCES = ["orb", "daily", "burst"]

# Short labels used in the combined tab's "found by" column.
SOURCE_TAGS = {"orb": "ORB", "daily": "DAILY", "burst": "BURST"}

# A scan older than this many hours gets a stale warning on its tab. The
# intraday check gets a much shorter fuse: at breakfast it will be holding
# yesterday afternoon's picture, and that needs saying plainly.
STALE_HOURS = 36
STALE_HOURS_BY_SCAN = {"burst_check": 8}

UNIVERSE_URLS = [
    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_scan(scan_id):
    """Read one scan's JSON. Returns a 'missing' stub if it was never written."""
    path = os.path.join(DATA_DIR, "%s.json" % scan_id)
    if not os.path.exists(path):
        return {
            "scan_id": scan_id, "label": scan_id, "status": "missing",
            "generated_utc": "", "count": 0, "columns": [], "meta": {},
            "rows": [], "error": "No %s has been written yet." % path,
        }
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        return {
            "scan_id": scan_id, "label": scan_id, "status": "failed",
            "generated_utc": "", "count": 0, "columns": [], "meta": {},
            "rows": [], "error": "Could not read %s: %s" % (path, exc),
        }
    data.setdefault("rows", [])
    data.setdefault("columns", [])
    data.setdefault("meta", {})
    data.setdefault("status", "ok")
    return data


def hours_old(stamp):
    """Age of an ISO-ish UTC timestamp in hours, or None if unparseable."""
    if not stamp:
        return None
    try:
        clean = stamp.replace("Z", "").replace("T", " ").strip()
        when = datetime.strptime(clean[:19], "%Y-%m-%d %H:%M:%S")
        when = when.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - when).total_seconds() / 3600.0
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Exchange lookup
# ---------------------------------------------------------------------------

def build_exchange_map():
    """Map every listed symbol to its TradingView exchange prefix.

    The burst and anticipation scans already tag their rows, but the ORB
    screener does not, so this fills the gaps. Uses the same NASDAQ Trader
    files and the same rules the scans use, to keep the prefixes consistent.

    Returns an empty dict on any failure -- the page still builds, symbols
    just go into the .txt files without a prefix.
    """
    mapping = {}
    for url in UNIVERSE_URLS:
        is_nasdaq_file = "nasdaqlisted" in url
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "build-scans/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            print("[build_scans] exchange lookup failed for %s: %s" % (url, exc))
            continue

        lines = [l for l in text.splitlines() if "|" in l]
        if not lines:
            continue
        header = lines[0].split("|")
        idx = {name: i for i, name in enumerate(header)}
        sym_col = "Symbol" if "Symbol" in idx else "ACT Symbol"
        if sym_col not in idx:
            continue

        for line in lines[1:]:
            f = line.split("|")
            if len(f) < len(header) or f[0].startswith("File Creation"):
                continue
            sym = f[idx[sym_col]].strip().upper()
            if not sym:
                continue
            if is_nasdaq_file:
                mapping[sym] = "NASDAQ"
            else:
                exch = f[idx["Exchange"]].strip() if "Exchange" in idx else ""
                mapping[sym] = "NYSE" if exch == "N" else "AMEX"

    print("[build_scans] exchange map: %d symbols" % len(mapping))
    return mapping


def fill_exchanges(scan, exchange_map):
    """Give every row an exchange prefix where one can be found."""
    filled = 0
    for row in scan.get("rows", []):
        if not row.get("exchange"):
            found = exchange_map.get((row.get("symbol") or "").upper(), "")
            if found:
                row["exchange"] = found
                filled += 1
    if filled:
        print("[build_scans] %s: filled %d exchanges" % (scan["scan_id"], filled))
    return scan


# ---------------------------------------------------------------------------
# Combined list
# ---------------------------------------------------------------------------

def build_combined(scans):
    """Merge ORB, Daily and Burst into one deduplicated list.

    A symbol found by more than one scan is the strongest signal the system
    produces, so the merged list records which scans found each name and
    sorts the multi-scan names to the top.
    """
    merged = {}
    for scan_id in COMBINED_SOURCES:
        scan = scans.get(scan_id)
        if not scan or scan.get("status") not in ("ok", "empty"):
            continue
        tag = SOURCE_TAGS.get(scan_id, scan_id.upper())
        for row in scan.get("rows", []):
            sym = (row.get("symbol") or "").upper()
            if not sym:
                continue
            if sym not in merged:
                merged[sym] = {
                    "symbol": sym,
                    "exchange": row.get("exchange", ""),
                    "close": row.get("close"),
                    "extra": {"found_by": [tag]},
                }
            else:
                entry = merged[sym]
                if tag not in entry["extra"]["found_by"]:
                    entry["extra"]["found_by"].append(tag)
                if not entry["exchange"] and row.get("exchange"):
                    entry["exchange"] = row["exchange"]
                if entry["close"] is None and row.get("close") is not None:
                    entry["close"] = row["close"]

    rows = list(merged.values())
    # Most confluence first, then alphabetical so the order is stable.
    rows.sort(key=lambda r: (-len(r["extra"]["found_by"]), r["symbol"]))
    for r in rows:
        r["extra"]["scans"] = len(r["extra"]["found_by"])
        r["extra"]["found_by"] = " + ".join(r["extra"]["found_by"])

    present = [SOURCE_TAGS[s] for s in COMBINED_SOURCES
               if scans.get(s, {}).get("status") in ("ok", "empty")]
    missing = [SOURCE_TAGS[s] for s in COMBINED_SOURCES
               if scans.get(s, {}).get("status") not in ("ok", "empty")]

    return {
        "scan_id": "combined",
        "label": "Combined watchlist",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "ok" if rows else "empty",
        "count": len(rows),
        "columns": ["found_by", "scans"],
        "meta": {
            "sources": ", ".join(present) if present else "none",
            "sort": "most scans first, then A-Z",
        },
        "rows": rows,
        "incomplete": missing,
    }


# ---------------------------------------------------------------------------
# TradingView export
# ---------------------------------------------------------------------------

def tv_symbol(row):
    sym = (row.get("symbol") or "").strip().upper()
    exch = (row.get("exchange") or "").strip().upper()
    return "%s:%s" % (exch, sym) if exch else sym


def write_txt(scan_id, scan):
    os.makedirs(OUT_DIR, exist_ok=True)
    names = [tv_symbol(r) for r in scan.get("rows", []) if r.get("symbol")]
    path = os.path.join(OUT_DIR, "scans_%s.txt" % scan_id)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(names))
        if names:
            fh.write("\n")
    return path, names


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def esc(value):
    if value is None:
        return ""
    return html.escape(str(value))


def fmt_cell(value):
    if value is None or value == "":
        return "&mdash;"
    if isinstance(value, float):
        return esc("%g" % value)
    return esc(value)


def status_note(scan):
    """Plain-English line about whether this scan can be trusted."""
    sid = scan.get("scan_id", "")
    status = scan.get("status", "ok")
    age = hours_old(scan.get("generated_utc"))
    intraday = sid in INTRADAY
    limit = STALE_HOURS_BY_SCAN.get(sid, STALE_HOURS)

    if status == "missing":
        if intraday:
            return "warn", ("Has not run yet today. This tab fills in once "
                            "the US market has been open an hour.")
        return "bad", "Has not run yet. No results file was found."
    if status == "failed":
        return "bad", "Failed: %s" % scan.get("error", "unknown error")
    if age is not None and age > limit:
        if intraday:
            return "warn", ("From an earlier session, %.0f hours ago. This "
                            "tab refreshes an hour after the US open." % age)
        return "warn", ("Last ran %.0f hours ago. These results are out of "
                        "date." % age)
    if status == "empty":
        if intraday:
            return "good", ("Checked, and none of last night's burst names "
                            "are bursting again yet.")
        return "warn", "Ran successfully but found nothing today."
    return "good", "Ran successfully."


def render_table(scan):
    rows = scan.get("rows", [])
    if not rows:
        return '<p class="empty">Nothing to show.</p>'

    cols = scan.get("columns", [])
    head = ['<th class="sym">Symbol</th>', "<th>Exch</th>",
            '<th class="num">Close</th>']
    for c in cols:
        cls = "num" if c not in ("quals", "found_by", "trend200",
                                 "burst_date", "setup_date", "triangle_date",
                                 "status", "volume_ok", "range_ok") else ""
        head.append('<th class="%s">%s</th>' % (cls, esc(c)))

    body = []
    for r in rows:
        cells = ['<td class="sym">%s</td>' % esc(r.get("symbol")),
                 "<td>%s</td>" % (esc(r.get("exchange")) or "&mdash;"),
                 '<td class="num">%s</td>' % fmt_cell(r.get("close"))]
        extra = r.get("extra", {})
        for c in cols:
            val = extra.get(c)
            if c == "found_by":
                n = extra.get("scans", 1)
                cls = "tag multi" if n > 1 else "tag"
                cells.append('<td><span class="%s">%s</span></td>'
                             % (cls, esc(val)))
            else:
                cls = "num" if c not in ("quals", "trend200", "burst_date",
                                         "setup_date", "triangle_date",
                                         "status", "volume_ok",
                                         "range_ok") else ""
                cells.append('<td class="%s">%s</td>' % (cls, fmt_cell(val)))
        body.append("<tr>%s</tr>" % "".join(cells))

    return ('<table><thead><tr>%s</tr></thead><tbody>%s</tbody></table>'
            % ("".join(head), "".join(body)))


def render_panel(scan_id, scan, names):
    tone, note = status_note(scan)
    meta = scan.get("meta", {})
    bits = []
    for key in ("sort", "universe", "found", "scanned", "cap", "errors",
                "burst_date", "setup_date", "sources"):
        if key in meta and meta[key] not in ("", None):
            bits.append("%s: %s" % (esc(key.replace("_", " ")), esc(meta[key])))

    incomplete = scan.get("incomplete") or []
    warn = ""
    if incomplete:
        warn = ('<p class="incomplete">Built without %s &mdash; '
                'that scan did not produce results, so names it would have '
                'contributed are missing from this list.</p>'
                % esc(", ".join(incomplete)))

    stamp = esc(scan.get("generated_utc", "")) or "never"
    listing = ",".join(names)

    return """<section class="panel" id="panel-{sid}" hidden>
  <div class="status {tone}">
    <span class="dot"></span>
    <span class="note">{note}</span>
    <span class="stamp">{stamp}</span>
  </div>
  {warn}
  <div class="bar">
    <span class="count"><strong>{count}</strong> {word}</span>
    <a class="btn" href="scans_{sid}.txt" download>Download .txt</a>
    <button class="btn ghost" type="button" data-copy="tv-{sid}">Copy tickers</button>
    <span class="meta">{meta}</span>
  </div>
  <textarea id="tv-{sid}" readonly onclick="this.select()">{listing}</textarea>
  {table}
</section>""".format(
        sid=esc(scan_id), tone=tone, note=esc(note), stamp=stamp, warn=warn,
        count=len(scan.get("rows", [])),
        word="symbol" if len(scan.get("rows", [])) == 1 else "symbols",
        meta=" &middot; ".join(bits), listing=esc(listing),
        table=render_table(scan))


def render_chip(scan_id, label, scan):
    tone, _ = status_note(scan)
    live = " live" if scan_id in INTRADAY else ""
    return ('<button class="tab {tone}{live}" type="button" data-tab="{sid}">'
            '<span class="dot"></span>{label}'
            '<span class="n">{n}</span></button>').format(
        tone=tone, live=live, sid=esc(scan_id), label=esc(label),
        n=len(scan.get("rows", [])))


CSS = """
:root{
  --bg:#0d1117; --panel:#11161d; --line:#21262d; --line2:#2d333b;
  --ink:#e6edf3; --dim:#8b949e; --accent:#58a6ff;
  --good:#3fb950; --warn:#d29922; --bad:#f85149;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
body{margin:0;padding:20px;background:var(--bg);color:var(--ink);
  font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
header{display:flex;flex-wrap:wrap;align-items:baseline;gap:12px;
  margin-bottom:4px}
h1{font-size:19px;margin:0;letter-spacing:-.01em}
header .built{color:var(--dim);font-size:12px;font-family:var(--mono)}
header nav{margin-left:auto;display:flex;gap:14px}
header nav a{color:var(--accent);text-decoration:none;font-size:13px}
header nav a:hover{text-decoration:underline}
.lede{color:var(--dim);font-size:13px;margin:0 0 18px}

.tabs{display:flex;flex-wrap:wrap;gap:6px;border-bottom:1px solid var(--line);
  padding-bottom:0;margin-bottom:0}
.tab{appearance:none;background:transparent;border:1px solid transparent;
  border-bottom:none;border-radius:8px 8px 0 0;color:var(--dim);
  font:inherit;font-weight:600;padding:9px 14px;cursor:pointer;
  display:flex;align-items:center;gap:8px;position:relative;top:1px}
.tab:hover{color:var(--ink)}
.tab[aria-selected="true"]{background:var(--panel);color:var(--ink);
  border-color:var(--line);border-bottom:1px solid var(--panel)}
.tab .n{font-family:var(--mono);font-size:11px;font-weight:400;
  background:var(--line);color:var(--dim);border-radius:9px;padding:1px 7px}
.tab[aria-selected="true"] .n{background:var(--line2);color:var(--ink)}
.tab:focus-visible,.btn:focus-visible{outline:2px solid var(--accent);
  outline-offset:2px}

.dot{width:7px;height:7px;border-radius:50%;flex:none;background:var(--dim)}
.good .dot{background:var(--good)}
.warn .dot{background:var(--warn)}
.bad .dot{background:var(--bad)}

.panel{background:var(--panel);border:1px solid var(--line);border-top:none;
  border-radius:0 0 10px 10px;padding:16px}
.panel[hidden]{display:none}

.status{display:flex;align-items:center;gap:9px;font-size:13px;
  padding:9px 12px;border-radius:7px;margin-bottom:14px;
  border:1px solid var(--line2)}
.status.good{color:var(--dim)}
.status.warn{color:var(--warn);border-color:#5c4708;background:#1d1804}
.status.bad{color:var(--bad);border-color:#5c1a17;background:#1d0f0e}
.status .stamp{margin-left:auto;font-family:var(--mono);font-size:11px;
  color:var(--dim)}
.incomplete{color:var(--warn);font-size:13px;margin:-6px 0 14px}

.bar{display:flex;flex-wrap:wrap;align-items:center;gap:10px;
  margin-bottom:10px}
.bar .count{font-size:13px;color:var(--dim)}
.bar .count strong{color:var(--ink);font-family:var(--mono);font-size:15px}
.bar .meta{color:var(--dim);font-size:12px;flex-basis:100%;
  font-family:var(--mono)}
.btn{display:inline-block;border:1px solid transparent;border-radius:7px;
  background:var(--accent);color:#04101f;font:inherit;font-weight:600;
  font-size:13px;padding:7px 13px;cursor:pointer;text-decoration:none}
.btn:hover{filter:brightness(1.1)}
.btn.ghost{background:transparent;color:var(--ink);border-color:var(--line2)}
.btn.ghost:hover{background:var(--line)}

textarea{width:100%;height:44px;margin-bottom:14px;background:var(--bg);
  color:var(--dim);border:1px solid var(--line);border-radius:7px;
  padding:8px 10px;font-family:var(--mono);font-size:11px;resize:vertical}

table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:7px 11px;text-align:left;border-bottom:1px solid var(--line);
  white-space:nowrap}
th{color:var(--dim);font-size:11px;font-weight:600;text-transform:uppercase;
  letter-spacing:.04em;cursor:pointer;user-select:none;position:sticky;top:0;
  background:var(--panel)}
th:hover{color:var(--ink)}
th::after{content:"";opacity:.45;font-size:9px;margin-left:5px}
th.asc::after{content:"\\25B2"}
th.desc::after{content:"\\25BC"}
td.num,th.num{text-align:right;font-family:var(--mono)}
td.sym{font-family:var(--mono);font-weight:700}
tbody tr:hover{background:#161c24}
.tag{font-family:var(--mono);font-size:11px;color:var(--dim);
  border:1px solid var(--line2);border-radius:5px;padding:1px 6px}
.tag.multi{color:var(--good);border-color:#1f5c2e;background:#08160d;
  font-weight:700}
.empty{color:var(--dim);padding:22px 0;text-align:center}
.tab.live{margin-left:auto}
.tab.live::before{content:"intraday";font-size:9px;letter-spacing:.08em;
  text-transform:uppercase;color:var(--dim);border:1px solid var(--line2);
  border-radius:4px;padding:1px 5px}

@media (max-width:640px){
  body{padding:12px}
  .tab{padding:8px 10px;font-size:13px}
  th,td{padding:6px 8px}
}
"""

JS = """
(function(){
  var tabs=[].slice.call(document.querySelectorAll('.tab'));
  function show(id){
    tabs.forEach(function(t){
      var on=t.dataset.tab===id;
      t.setAttribute('aria-selected',on?'true':'false');
      var p=document.getElementById('panel-'+t.dataset.tab);
      if(p)p.hidden=!on;
    });
    try{location.hash=id;}catch(e){}
  }
  tabs.forEach(function(t){
    t.addEventListener('click',function(){show(t.dataset.tab);});
  });
  var start=(location.hash||'').replace('#','');
  show(tabs.some(function(t){return t.dataset.tab===start;})?start:tabs[0].dataset.tab);

  document.addEventListener('click',function(e){
    var b=e.target.closest('[data-copy]');
    if(!b)return;
    var ta=document.getElementById(b.dataset.copy);
    if(!ta)return;
    var done=function(){var o=b.textContent;b.textContent='Copied';
      setTimeout(function(){b.textContent=o;},1400);};
    if(navigator.clipboard){navigator.clipboard.writeText(ta.value).then(done,function(){
      ta.select();document.execCommand('copy');done();});}
    else{ta.select();document.execCommand('copy');done();}
  });

  document.querySelectorAll('table').forEach(function(tbl){
    tbl.querySelectorAll('th').forEach(function(th,i){
      th.addEventListener('click',function(){
        var body=tbl.tBodies[0];
        var rows=[].slice.call(body.rows);
        var desc=!th.classList.contains('desc');
        tbl.querySelectorAll('th').forEach(function(o){
          o.classList.remove('asc','desc');});
        th.classList.add(desc?'desc':'asc');
        rows.sort(function(a,b){
          var x=a.cells[i].textContent.trim(),y=b.cells[i].textContent.trim();
          var nx=parseFloat(x.replace(/[^0-9.eE+-]/g,'')),
              ny=parseFloat(y.replace(/[^0-9.eE+-]/g,''));
          var num=!isNaN(nx)&&!isNaN(ny)&&x!==''&&y!=='';
          var c=num?(nx-ny):x.localeCompare(y);
          return desc?-c:c;
        });
        rows.forEach(function(r){body.appendChild(r);});
      });
    });
  });
})();
"""


def build_page(scans, combined, listings):
    order = [("orb", "ORB"), ("daily", "Daily"), ("burst", "Burst"),
             ("combined", "Combined"), ("anticipation", "Anticipation"),
             ("burst_check", "Second burst")]
    all_scans = dict(scans)
    all_scans["combined"] = combined

    chips = "".join(render_chip(sid, label, all_scans[sid])
                    for sid, label in order)
    panels = "".join(render_panel(sid, all_scans[sid], listings.get(sid, []))
                     for sid, _ in order)
    built = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return """<!DOCTYPE html>
<html lang="en-GB"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Overnight scans</title>
<style>{css}</style></head><body>
<header>
  <h1>Overnight scans</h1>
  <span class="built">built {built}</span>
  <nav>
    <a href="index.html">ORB watchlist</a>
    <a href="setups.html">Anticipation intraday</a>
  </nav>
</header>
<p class="lede">Five scans of last night's close, plus one live check that
runs during the US session. Every tab exports a TradingView import file;
Combined merges ORB, Daily and Burst with duplicates removed.</p>
<div class="tabs" role="tablist">{chips}</div>
{panels}
<script>{js}</script>
</body></html>""".format(css=CSS, js=JS, built=built, chips=chips, panels=panels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    scans = {sid: load_scan(sid) for sid, _ in SCANS}

    exchange_map = build_exchange_map()
    if exchange_map:
        for scan in scans.values():
            fill_exchanges(scan, exchange_map)

    combined = build_combined(scans)

    listings = {}
    for sid in list(scans) + ["combined"]:
        scan = combined if sid == "combined" else scans[sid]
        path, names = write_txt(sid, scan)
        listings[sid] = names
        print("[build_scans] %s: %d symbols -> %s" % (sid, len(names), path))

    page = os.path.join(OUT_DIR, "scans.html")
    with open(page, "w", encoding="utf-8") as fh:
        fh.write(build_page(scans, combined, listings))
    print("[build_scans] wrote %s" % page)

    broken = [s["scan_id"] for s in scans.values()
              if s.get("status") in ("failed", "missing")]
    if broken:
        print("[build_scans] NOTE: no usable results from: %s"
              % ", ".join(broken))
    return 0


if __name__ == "__main__":
    sys.exit(main())
