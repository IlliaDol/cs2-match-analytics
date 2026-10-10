"""Fill the native names in data/player_names.csv from Liquipedia.

Why a crawl instead of a transliteration table: the dataset stores only a Latin form,
and for Ukrainian players that form is frequently the *Russian* variant
(`Aleksandr Kostyliev` for `Олександр Олегович Костилєв`). A Liquipedia player page
carries the name in the player's own script (`|name=`) plus a romanisation
(`|romanized_name=`), so nothing here is invented — every value is copied off the page.
The page's `|country=` is compared with the dataset's country code, which is what
catches a nickname that belongs to a different person.

Politeness: one request every 2.2 s with a descriptive User-Agent, and every response
is cached under data/interim/liquipedia/ (git-ignored), so re-runs cost no traffic.
Idempotent: rows that already have a native name are skipped, and the columns Illia
edits himself (`corrected_latin`, `note`) are never overwritten.

    .venv/Scripts/python.exe scripts/fetch_player_names.py            # fill what is empty
    .venv/Scripts/python.exe scripts/fetch_player_names.py --limit 5  # try a few first
    .venv/Scripts/python.exe scripts/fetch_player_names.py --refresh  # ignore the cache
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
SHEET = REPO / "data" / "player_names.csv"
CACHE = REPO / "data" / "interim" / "liquipedia"
UA = "IlliaDol-cs2-analytics/1.0 (personal dataset project; github.com/IlliaDol)"
DELAY_S = 2.2
BASE = "https://liquipedia.net/counterstrike/"

# this script prints Cyrillic names; a cp1252 console would raise instead of printing them
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

FIELDS = ["country", "language", "nickname", "dataset_string", "latin_name_in_data",
          "maps", "native_name", "liquipedia_romanized", "liquipedia_country",
          "corrected_latin", "note"]
FIELD_RE = {k: re.compile(r"^\|\s*" + k + r"\s*=\s*(.+?)\s*$", re.M)
            for k in ("name", "romanized_name", "country")}
REDIRECT_RE = re.compile(r"^#REDIRECT\s*\[\[([^\]]+)\]\]", re.I | re.M)


def safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", text)[:80] or "page"


def fetch_wikitext(title: str, refresh: bool = False) -> tuple[str | None, str]:
    """Raw wikitext of a Liquipedia player page, cached on disk.

    Returns (wikitext or None, status) where status explains a miss.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"{safe_slug(title)}.txt"
    if cached.exists() and not refresh:
        text = cached.read_text(encoding="utf-8")
        return (None, text) if text.startswith("\x00MISS") else (text, "cache")
    url = BASE + urllib.parse.quote(title.replace(" ", "_")) + "?action=raw"
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            text = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        cached.write_text(f"\x00MISS{exc.code}", encoding="utf-8")
        return None, f"http {exc.code}"
    except Exception as exc:                                  # noqa: BLE001 - report, never crash
        return None, f"{type(exc).__name__}: {exc}"
    time.sleep(DELAY_S)                                       # only after real traffic
    # a redirect page holds no data: follow it once
    redirect = REDIRECT_RE.search(text)
    if redirect and len(text) < 400:
        return fetch_wikitext(redirect.group(1).strip(), refresh)
    cached.write_text(text, encoding="utf-8")
    return text, "fetched"


def field(text: str, key: str) -> str:
    found = FIELD_RE[key].search(text)
    if not found:
        return ""
    value = found.group(1).strip()
    # some pages wrap the value in a template or add a footnote
    value = re.sub(r"<ref[^>]*>.*?</ref>", "", value, flags=re.S).strip()
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, help="only the first N rows that need filling")
    parser.add_argument("--refresh", action="store_true", help="ignore the on-disk cache")
    parser.add_argument("--sheet", default=str(SHEET), help="path to the review sheet")
    args = parser.parse_args(argv)

    path = Path(args.sheet)
    with path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:                                          # tolerate older sheets
        for column in FIELDS:
            row.setdefault(column, "")

    todo = [r for r in rows if not (r.get("native_name") or "").strip()]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(rows)} players in the sheet, {len(todo)} still need a native name")

    filled = missed = mismatched = 0
    for index, row in enumerate(todo, 1):
        nick = row["nickname"] or row["dataset_string"].split(" (")[0]
        text, status = fetch_wikitext(nick, args.refresh)
        if text is None:
            row["note"] = (row.get("note") or "") + f" [liquipedia: {status}]"
            missed += 1
            print(f"  {index:>3}/{len(todo)}  {nick:<18} MISS ({status})")
            continue
        native = field(text, "name")
        roman = field(text, "romanized_name")
        page_country = field(text, "country")
        row["native_name"] = native
        row["liquipedia_romanized"] = roman
        row["liquipedia_country"] = page_country
        if native:
            filled += 1
        note = row.get("note") or ""
        want = (row.get("country") or "").upper()
        lookup = {"ukraine": "UA", "russia": "RU", "kazakhstan": "KZ", "belarus": "BY",
                  "mongolia": "MN", "uzbekistan": "UZ", "kyrgyzstan": "KG",
                  "azerbaijan": "AZ", "estonia": "EE", "latvia": "LV", "lithuania": "LT"}
        got = lookup.get(page_country.casefold(), page_country.upper()[:2])
        if want and got and got != want:
            note = (note + f" [country: sheet {want} vs page {got} —"
                           " check this is the right player]").strip()
            mismatched += 1
        row["note"] = note
        print(f"  {index:>3}/{len(todo)}  {nick:<18} {native or '(no |name= on the page)'}"
              + (f"  [{status}]" if status != "fetched" else ""))

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})
    print(f"\n{native_count(rows)}/{len(rows)} rows now carry a native name "
          f"({filled} filled now, {missed} pages missing, {mismatched} country mismatches)")
    print(f"wrote {path}")
    return 0


def native_count(rows: list[dict[str, str]]) -> int:
    return sum(1 for r in rows if (r.get("native_name") or "").strip())


if __name__ == "__main__":
    sys.exit(main())
