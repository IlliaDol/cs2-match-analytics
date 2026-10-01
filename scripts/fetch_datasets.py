"""Fetch the extra CS2 datasets listed in docs/DATA-EXPANSION.md.

Needs a Kaggle API token at ``%USERPROFILE%\\.kaggle\\kaggle.json``
(kaggle.com → Account → API → *Create New API Token*). The script checks that first
and refuses with the exact instructions instead of failing deep inside the client.

Downloads land in ``data/raw/new/`` so the existing v1 file set is never overwritten —
the current pipeline keeps working unchanged until the new files are reviewed.

    .venv/Scripts/python.exe scripts/fetch_datasets.py --list
    .venv/Scripts/python.exe scripts/fetch_datasets.py --all
    .venv/Scripts/python.exe scripts/fetch_datasets.py --dataset ektarr/counter-strike-pro-matches
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "data" / "raw" / "new"
TOKEN = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"

# registry: slug -> what it adds (kept in sync with docs/DATA-EXPANSION.md)
DATASETS: dict[str, str] = {
    "ektarr/counter-strike-pro-matches":
        "Tier A refresh — our main source; newer months at the same schema",
    "eupeutro/blast-rivals-2026-cs2-match-statistics":
        "Tier B — HLTV Rating 3.0 + Swing, to validate our metrics",
}


def token_status() -> tuple[bool, str]:
    if not TOKEN.exists():
        return False, (
            f"no Kaggle token at {TOKEN}\n"
            "  1. open https://www.kaggle.com/settings/account\n"
            "  2. 'Create New API Token' → downloads kaggle.json\n"
            f"  3. move it to {TOKEN.parent}\\  (the folder, not the repo)"
        )
    try:
        blob = json.loads(TOKEN.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        return False, f"{TOKEN} is not valid JSON: {err}"
    if not blob.get("username") or not blob.get("key"):
        return False, f"{TOKEN} lacks a username/key pair"
    return True, f"token for user '{blob['username']}'"


def download(slug: str, force: bool = False) -> int:
    target = DEST / slug.split("/")[-1]
    if target.exists() and any(target.iterdir()) and not force:
        print(f"  = {slug}: already in {target.relative_to(REPO)} (use --force to refetch)")
        return 0
    target.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ {slug} → {target.relative_to(REPO)}")
    result = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download", "-d", slug, "-p", str(target),
         "--unzip"],
        check=False,
    )
    if result.returncode != 0:
        print(f"  ✗ {slug}: kaggle client exited {result.returncode}")
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="show the registry and stop")
    parser.add_argument("--all", action="store_true", help="fetch every registered dataset")
    parser.add_argument("--dataset", action="append", default=[], metavar="OWNER/SLUG",
                        help="fetch one dataset (repeatable)")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--check-token", action="store_true", help="only verify the token")
    args = parser.parse_args(argv)

    print(f"destination: {DEST.relative_to(REPO)}  (the v1 files in data/raw/ are never touched)")
    if args.list:
        for slug, note in DATASETS.items():
            print(f"  {slug}\n      {note}")
        return 0

    ok, message = token_status()
    print(f"token: {message}")
    if args.check_token:
        return 0 if ok else 1
    if not ok:
        print("\ncannot download without a token — nothing was changed.")
        return 1

    chosen = list(DATASETS) if args.all else args.dataset
    if not chosen:
        parser.error("nothing to do: pass --all or --dataset OWNER/SLUG (or --list)")
    unknown = [slug for slug in chosen if slug not in DATASETS]
    if unknown:
        print("not in the registry: " + ", ".join(unknown)
              + " — add it here and to docs/DATA-EXPANSION.md")
        return 2
    failures = sum(download(slug, force=args.force) != 0 for slug in chosen)
    print(f"\n{len(chosen) - failures}/{len(chosen)} dataset(s) fetched into "
          f"{DEST.relative_to(REPO)}")
    print("next: compare schemas against data/raw/*.csv before wiring them into the pipeline")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
