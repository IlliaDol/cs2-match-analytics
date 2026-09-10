# M9 BUILD SPEC — Big Data: the same pipeline in PySpark  · `DE MLE`

**Environment:** Colab (free tier). No local Java/Spark setup on Windows — do NOT try.
The deliverable is a committed notebook + a timing table, not a cluster.

---

## §1 `notebooks/05_spark_features.ipynb`

Runs ENTIRELY in Colab. First cell:
```python
!pip -q install pyspark==3.5.*
```
Then:

1. **Load** `outputs/series_clean.csv` (upload it or `!wget` the raw GitHub URL — the repo
   is public, so `https://raw.githubusercontent.com/IlliaDol/cs2-match-analytics/main/outputs/series_clean.csv`
   works).
2. **Scale-up trick:** `.unionAll` the 9,922-row frame onto itself 200× with a synthetic
   `replicate_id` column → ~2M rows. Now timing differences are real, not noise.
3. **Reimplement M6's rolling form in Spark** with window functions:
   `Window.partitionBy("team").orderBy("datetime").rowsBetween(-4, -1)` — the Spark twin of
   `rolling_form`. Compare its output on the ORIGINAL 9,922 rows against the pandas result
   (assert equality within 1e-9 for the overlap) — that is your correctness proof.
4. **Timing table** → `outputs/m9_spark_vs_pandas.csv` with columns
   `[scale_rows, pandas_sec, spark_sec, speedup]` for at least 3 scales
   (9,922 / 100k / 500k rows). Include the honest result: at 10k rows pandas usually WINS
   (JVM startup); Spark wins at scale. Say that in a markdown cell.
5. **Batch vs streaming** — one markdown cell, 4–6 sentences, using THIS project:
   nightly results batch (HLTV dump → parquet → rating update) vs live round-by-round
   stream (what you'd need for in-play odds). Name one thing that breaks in each.

## §2 Repo artifacts
- `notebooks/05_spark_features.ipynb` committed WITH outputs saved (Colab: File → Save).
- `outputs/m9_spark_vs_pandas.csv` committed.
- Add to `docs/DECISIONS.md`: "Spark via Colab, not local — Windows + no JVM, and the
  dataset doesn't justify a cluster; the notebook is the artifact."

## §3 Tests (`tests/test_m9_artifacts.py`, skipif files missing)
- notebook exists, is valid JSON, contains a cell with `SparkSession` and one with
  `Window.partitionBy`
- timing CSV: ≥ 3 rows, monotone `scale_rows`, all times > 0, `speedup == pandas/spark`
  within 1e-6

## Checkpoint
1. Explain in your own words where the shuffle happens in the rolling-form job and why
   partitioning by team matters (what happens to the DAG if you don't).
2. Why does pandas beat Spark at 10k rows? Name two specific overheads.
3. What would change if the source were a Kafka topic instead of a CSV?

## Appendix — hints
- Spark window syntax:
  `w = Window.partitionBy("team").orderBy(col("datetime").cast("long")).rowsBetween(-4, -1)`
  then `F.avg("won").over(w)`.
- Spark needs `spark.conf.set("spark.sql.shuffle.partitions", "8")` on Colab or it spawns
  200 partitions and looks artificially slow — say so in the notebook.
- If `!wget` from raw.githubusercontent fails, use `files.upload()` from
  `google.colab` and document the path in a comment.
