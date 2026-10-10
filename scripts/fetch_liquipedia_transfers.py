"""Fetch the roster/transfer history from Liquipedia's public wiki — the missing dates.

Why this exists: `roster_stability_diff` and `standin_diff` are the two strongest
features in the model, and they are built from per-map line-ups. But nothing in the match
data says *when* a player joined or left a team — a stand-in looks identical to a
permanent member. Liquipedia's transfer pages carry exactly that, one structured template
per move:

    {{Transfer Row|name=dev1ce|flag=dk|name2=Magisk|flag2=dk|team1=astralis|date=2025-12-31}}
    {{Transfer Row|name=Plopski|flag=se|team1=metizport|team2=metizport|role1=Inactive|...}}

name / flag / team1 (leaving) / team2 (joining) / role1 / role2 / **date**. Verified by
hand on 2026-09-15: ~237 rows for 2025-12, 451 for 2020-01, 147 for 2016-01, and 404 for
2014-01 — so the coverage is roughly 2015-07 onward, which spans both CS:GO and CS2 and
covers the 2020-2021 window where our *match* data has a hole.

Politeness: Liquipedia asks for at most one request every 2 seconds and a descriptive
User-Agent. This honours both, and caches every page under data/interim/ (git-ignored), so
re-running costs no traffic. Never raise the rate.

    .venv/Scripts/python.exe scripts/fetch_liquipedia_transfers.py --from 2015-07
    .venv/Scripts/python.exe scripts/fetch_liquipedia_transfers.py --months 2025-12 --dry
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data" / "interim" / "liquipedia_transfers"
OUT = REPO / "data" / "interim" / "roster_transfers.csv"
UA = "IlliaDol-cs2-analytics/1.0 (personal dataset project; github.com/IlliaDol)"
DELAY_S = 2.0                      # Liquipedia's documented limit: 1 request / 2 s
BASE = "https://liquipedia.net/counterstrike/Player_Transfers/"

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
ROW_RE = re.compile(r"\{\{Transfer Row\|(.*?)\}\}", re.S)
COLUMNS = ["date", "player", "flag", "from_team", "to_team", "role", "page"]


def parse_rows(wikitext: str, page: str) -> list[dict]:
    """Every transfer template on a page -> one row per person involved."""
    out: list[dict] = []
    for match in ROW_RE.finditer(wikitext):
        fields: dict[str, str] = {}
        for chunk in match.group(1).split("|"):
            if "=" not in chunk:
                continue
            key, _, value = chunk.partition("=")
            fields[key.strip()] = value.strip()
        date = fields.get("date", "")
        if not date:
            continue                                   # unusable without a date
        # one template can cover a whole line-up (name, name2 ... name5)
        for slot in ("", "2", "3", "4", "5"):
            player = fields.get(f"name{slot}")
            if not player:
                continue
            role = fields.get(f"role{slot}") or fields.get("role1", "")
            out.append({
                "date": date,
                "player": player,
                "flag": fields.get(f"flag{slot}", ""),
                "from_team": fields.get("team1", ""),
                "to_team": fields.get("team2", ""),
                "role": role,
                "page": page,
            })
    return out


def fetch(page: str, refresh: bool = False) -> str | None:
    """Raw wikitext of a transfers page, cached on disk."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / (page.replace("/", "_") + ".txt")
    if cached.exists() and not refresh:
        text = cached.read_text(encoding="utf-8")
        return None if text.startswith("\x00MISS") else text
    url = BASE + urllib.parse.quote(page.replace(" ", "_")) + "?action=raw"
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            text = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        cached.write_text(f"\x00MISS{exc.code}", encoding="utf-8")
        return None
    except Exception:                                  # noqa: BLE001 - report, never crash
        return None
    time.sleep(DELAY_S)                                # only after real traffic
    cached.write_text(text, encoding="utf-8")
    return text


def months_between(start: str, end: str) -> list[str]:
    start_year, start_month = (int(x) for x in start.split("-"))
    end_year, end_month = (int(x) for x in end.split("-"))
    out = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        out.append(f"{year}/{MONTHS[month - 1]}")
        month += 1
        if month > 12:
            year, month = year + 1, 1
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="start", default="2015-07",
                        help="first month, YYYY-MM (2014 pages do not exist)")
    parser.add_argument("--to", dest="end", default="",
                        help="last month, YYYY-MM (default: the current month)")
    parser.add_argument("--months", nargs="*", default=None,
                        help="explicit months to fetch, YYYY-MM (overrides --from/--to)")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--dry", action="store_true", help="print stats, write nothing")
    parser.add_argument("--refresh", action="store_true", help="ignore the cache")
    args = parser.parse_args(argv)

    if args.months:
        wanted = [f"{y}/{MONTHS[int(m) - 1]}" for y, m in
                  (m.split("-") for m in args.months)]
    else:
        end = args.end or time.strftime("%Y-%m")
        wanted = months_between(args.start, end)

    rows: list[dict] = []
    empty = 0
    for i, page in enumerate(wanted, 1):
        text = fetch(page, args.refresh)
        if text is None:
            empty += 1
            continue
        found = parse_rows(text, page)
        rows.extend(found)
        print(f"  {i:>3}/{len(wanted)}  {page:<28} {len(found):>4} rows")

    if not rows:
        print("nothing parsed — check the page titles, or that the cache is not all misses")
        return 1
    rows.sort(key=lambda r: (r["date"], r["player"]))
    dates = [r["date"] for r in rows]
    print(f"\ntotal       : {len(rows):,} transfer records")
    print(f"window      : {min(dates)} -> {max(dates)}")
    print(f"people      : {len({r['player'] for r in rows}):,}")
    print(f"pages empty : {empty}/{len(wanted)} (missing or not yet written)")

    if args.dry:
        print("\n--dry: nothing written")
        return 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
