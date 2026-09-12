# Nightly data-gated CI — what you must decide

The private Kaggle CSVs are git-ignored, so GitHub's normal CI cannot see them.
`.github/workflows/nightly-data.yml` fetches them from **one URL you control** at
03:23 UTC nightly, rebuilds the artifacts, runs the full data-gated suite, and
uploads `outputs/*.csv` as a CI artifact.

## What is already done

- Workflow file: `.github/workflows/nightly-data.yml` (schedule + `workflow_dispatch`).
- It rebuilds `interim`, `features_v1.parquet`, the trained model, and the drift
  report before `pytest`, so the skip guards (keyed on file existence) stop skipping
  and the suite actually exercises the data path.

## What you must do (pick ONE data source)

The workflow reads two repo secrets:

- `DATA_SOURCE_URL` — a URL pointing at a zip of your `data/raw/*.csv` files.
- `DATA_SOURCE_TOKEN` — optional bearer token, sent as `Authorization: Bearer ...`.

Pick one:

1. **GitHub release attachment (recommended).** Make a private release, attach
   `data_raw.zip`, set `DATA_SOURCE_URL` to `https://api.github.com/repos/<you>/<repo>/releases/assets/<asset_id>` and
   `DATA_SOURCE_TOKEN` to a fine-grained PAT with that one asset's read scope.
   Pros: free, no new accounts, token is least-privilege. Cons: asset id changes
   if you re-upload.
2. **Cloud object storage** (S3 / R2 / GCS bucket, private). Set `DATA_SOURCE_URL`
   to the object's pre-signed or public URL and the token to the access token.
   Pros: stable URL. Cons: another account + possible egress cost.
3. **Self-hosted runner + SSH** (if you want zero external data hosting): the
   CSVs stay on a machine you control and never leave it. Requires `docker` or a
   Linux box — you don't have Docker locally, so this is the heaviest option.

**Do NOT** commit the CSVs to the repo or use a public paste/transfer service —
Kaggle data licensing and GitHub's file-size limits both forbid the former, and
the latter breaks on both licensing and reliability.

## How to wire it

1. Google/GitHub → repo → **Settings → Secrets and variables → Actions**.
2. Add `DATA_SOURCE_URL` (and `DATA_SOURCE_TOKEN` if your host needs auth).
3. Run the workflow once by hand: **Actions → nightly-data → Run workflow**.

## Health signal (honesty guard)

The job **fails** when `DATA_SOURCE_URL` is unset instead of silently skipping —
that's deliberate. A green badge on a suite that skipped everything is a lie;
this workflow makes "green with data" vs "red without data" explicit.
