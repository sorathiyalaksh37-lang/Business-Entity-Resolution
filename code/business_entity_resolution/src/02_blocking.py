"""
02_blocking.py — Phase 2: Candidate Generation (Blocking)

Goal: For every Source 1 entity, find a manageable set of S2/S3 candidate records
that likely contains all true matches. Target recall >= 95%.

Strategy:
  1. Partition by country — never compare across countries
  2. Build TF-IDF index (unigrams + bigrams) on normalized name+address for S2+S3
  3. Stream S1 queries in tiny batches (200 rows) — avoids OOM crash
     Previous bug: BATCH_SIZE=5000 x 2M records x float32 = ~40 GB dense matrix!
     Fixed:        BATCH_SIZE=200  x 2M records x float32 = ~1.6 GB — safe
  4. Write candidates directly to disk per batch — no large RAM accumulation
  5. Measure recall on training ground truth
  6. Run for both train and test sets

Run from student_resource/:
    python3 code/business_entity_resolution/src/02_blocking.py

Outputs:
    output/train_candidate_pairs.tsv
    output/candidate_pairs.tsv
    code/business_entity_resolution/blocking_recall.txt
"""

import sys
import gc
import csv
import time
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))
from utils import normalize_name, normalize_address, extract_country_normalized

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).parents[3]   # student_resource/
TRAIN_DIR = ROOT / "dataset" / "train"
TEST_DIR  = ROOT / "dataset" / "test"
OUT_DIR   = ROOT / "output"
CODE_DIR  = Path(__file__).parents[1]   # code/business_entity_resolution/

OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K      = 100   # candidates per source (S2 + S3 separately) per S1 entity
BATCH_SIZE = 200   # S1 rows per batch — 200 x 2M x float32 = ~1.6 GB peak


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_tsv(path: Path) -> pd.DataFrame:
    print(f"  Loading {path.name}...", end=" ", flush=True)
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    print(f"{len(df):,} rows")
    return df


def make_text(df: pd.DataFrame) -> pd.Series:
    """name (x2 for weight) + address → single TF-IDF document."""
    n = df["business_name"].apply(normalize_name)
    a = df["business_address"].apply(normalize_address)
    return n + " " + n + " " + a


def add_country_norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_country"] = df["country"].apply(extract_country_normalized)
    return df


def build_index(texts: pd.Series, max_features: int = 150_000):
    """Fit TF-IDF + L2-normalize. Returns (vectorizer, sparse matrix)."""
    print(f"    TF-IDF on {len(texts):,} records...", end=" ", flush=True)
    t0 = time.time()
    vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        max_features=max_features, sublinear_tf=True, min_df=2,
    )
    mat = normalize(vec.fit_transform(texts), norm="l2", copy=False)
    print(f"shape={mat.shape} in {time.time()-t0:.1f}s")
    return vec, mat


# ---------------------------------------------------------------------------
# Memory-safe streaming search
# ---------------------------------------------------------------------------

def stream_search(s1_mat, s1_ids, s2_mat, s2_ids, s3_mat, s3_ids, writer):
    """
    Process S1 in batches of BATCH_SIZE.
    For each batch: compute cosine scores vs S2 and S3, pick top-K each,
    merge, deduplicate, write one TSV row — then free the dense arrays.
    """
    n = s1_mat.shape[0]
    total_cands = 0

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch = s1_mat[start:end]

        # S2 scores — shape (batch, n_s2) — freed after argpartition
        sc2 = (batch @ s2_mat.T).toarray()
        k2 = min(TOP_K, sc2.shape[1])
        idx2 = np.argpartition(sc2, -k2, axis=1)[:, -k2:]

        # S3 scores — shape (batch, n_s3) — freed after argpartition
        sc3 = (batch @ s3_mat.T).toarray()
        k3 = min(TOP_K, sc3.shape[1])
        idx3 = np.argpartition(sc3, -k3, axis=1)[:, -k3:]

        for i in range(end - start):
            v2 = idx2[i][sc2[i, idx2[i]] > 0.0]
            v3 = idx3[i][sc3[i, idx3[i]] > 0.0]
            cands = list(s2_ids[v2]) + list(s3_ids[v3])
            seen = set()
            unique = [c for c in cands if not (c in seen or seen.add(c))]
            writer.writerow([s1_ids[start + i], ",".join(unique)])
            total_cands += len(unique)

        del sc2, sc3, idx2, idx3, batch
        gc.collect()

        if (start // BATCH_SIZE) % 100 == 0:
            pct = end / n * 100
            print(f"      [{pct:5.1f}%] {end:,}/{n:,} queries | {total_cands:,} cands", flush=True)

    return total_cands


# ---------------------------------------------------------------------------
# Main blocking routine per split
# ---------------------------------------------------------------------------

def run_blocking(s1, s2, s3, split_name, out_path):
    print(f"\n{'='*60}\n  BLOCKING: {split_name}\n{'='*60}")
    s1 = add_country_norm(s1)
    s2 = add_country_norm(s2)
    s3 = add_country_norm(s3)

    countries = sorted(s1["_country"].unique())
    print(f"  Countries: {countries}")

    total_rows = total_cands = 0

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        for country in countries:
            print(f"\n  --- {country.upper()} ---")
            s1c = s1[s1["_country"] == country].reset_index(drop=True)
            s2c = s2[s2["_country"] == country].reset_index(drop=True)
            s3c = s3[s3["_country"] == country].reset_index(drop=True)
            print(f"    S1:{len(s1c):,}  S2:{len(s2c):,}  S3:{len(s3c):,}")

            if len(s1c) == 0:
                continue

            # No S2/S3 for this country → singletons
            if len(s2c) == 0 and len(s3c) == 0:
                for eid in s1c["entity_id"]:
                    writer.writerow([eid, ""])
                total_rows += len(s1c)
                continue

            # Fit TF-IDF on S2+S3 combined (shared vocabulary)
            s23_text = pd.concat([make_text(s2c), make_text(s3c)], ignore_index=True)
            vec, s23_mat = build_index(s23_text)

            n_s2 = len(s2c)
            s2_mat = s23_mat[:n_s2]
            s3_mat = s23_mat[n_s2:]
            s2_ids = s2c["entity_id"].values
            s3_ids = s3c["entity_id"].values

            # Transform S1 with the same vocabulary
            print(f"    Transforming {len(s1c):,} S1 queries...", end=" ", flush=True)
            t0 = time.time()
            s1_mat = normalize(vec.transform(make_text(s1c)), norm="l2", copy=False)
            s1_ids = s1c["entity_id"].values
            print(f"done in {time.time()-t0:.1f}s")

            print(f"    Streaming top-{TOP_K} search (batch={BATCH_SIZE})...")
            nc = stream_search(s1_mat, s1_ids, s2_mat, s2_ids, s3_mat, s3_ids, writer)
            total_rows += len(s1c)
            total_cands += nc

            del vec, s23_mat, s1_mat, s2_mat, s3_mat
            gc.collect()

    avg = total_cands / total_rows if total_rows else 0
    print(f"\n  Rows:{total_rows:,}  Candidates:{total_cands:,}  Avg/entity:{avg:.1f}")
    print(f"  Saved -> {out_path}")


# ---------------------------------------------------------------------------
# Blocking recall measurement
# ---------------------------------------------------------------------------

def measure_recall(candidate_path: Path, gt: pd.DataFrame):
    print("\n  Measuring blocking recall...")

    # Stream candidate file into lookup dict
    cand_lookup = {}
    with open(candidate_path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            raw = row.get("candidate_entity_ids", "").strip()
            cand_lookup[row["source1_entity_id"]] = set(raw.split(",")) if raw else set()

    total_true = total_found = 0
    entity_recalls = []

    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        raw = str(row.get("matched_entity_ids", "")).strip()
        true_m = set(x.strip() for x in raw.split(",") if x.strip()) if raw and raw != "nan" else set()
        if not true_m:
            continue
        found = len(true_m & cand_lookup.get(s1_id, set()))
        total_true += len(true_m)
        total_found += found
        entity_recalls.append(found / len(true_m))

    overall = total_found / total_true if total_true else 0
    mean_er = float(np.mean(entity_recalls)) if entity_recalls else 0
    perfect  = sum(1 for r in entity_recalls if r == 1.0)

    report = "\n".join([
        "=" * 60,
        "  BLOCKING RECALL REPORT",
        "=" * 60,
        f"  True pairs         : {total_true:,}",
        f"  Found in candidates: {total_found:,}",
        f"  Overall recall     : {overall:.4f}  ({overall*100:.2f}%)",
        f"  Mean entity recall : {mean_er:.4f}",
        f"  Perfect recall ents: {perfect:,} / {len(entity_recalls):,}",
        f"  TOP_K={TOP_K}  BATCH_SIZE={BATCH_SIZE}",
    ])
    print("\n" + report)
    (CODE_DIR / "blocking_recall.txt").write_text(report, encoding="utf-8")

    if overall < 0.90:
        print("\n  ⚠️  Recall < 90% — increase TOP_K!")
    elif overall >= 0.95:
        print(f"\n  ✅ Excellent recall: {overall*100:.2f}%")
    else:
        print(f"\n  ℹ️  Recall {overall*100:.2f}% — acceptable, consider TOP_K=150 if time allows")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  PHASE 2: BLOCKING / CANDIDATE GENERATION")
    print(f"  TOP_K={TOP_K}  BATCH_SIZE={BATCH_SIZE}")
    print("=" * 60)

    # Load all data upfront
    print("\nLoading training files:")
    s1_tr = load_tsv(TRAIN_DIR / "train_source1.tsv")
    s2_tr = load_tsv(TRAIN_DIR / "train_source2.tsv")
    s3_tr = load_tsv(TRAIN_DIR / "train_source3.tsv")
    gt_tr = load_tsv(TRAIN_DIR / "train_ground_truth.tsv")

    print("\nLoading test files:")
    s1_te = load_tsv(TEST_DIR / "test_source1.tsv")
    s2_te = load_tsv(TEST_DIR / "test_source2.tsv")
    s3_te = load_tsv(TEST_DIR / "test_source3.tsv")

    # --- TRAIN blocking ---
    train_out = OUT_DIR / "train_candidate_pairs.tsv"
    run_blocking(s1_tr, s2_tr, s3_tr, "TRAIN", train_out)
    measure_recall(train_out, gt_tr)

    # Free train data before test
    del s1_tr, s2_tr, s3_tr, gt_tr
    gc.collect()

    # --- TEST blocking ---
    test_out = OUT_DIR / "candidate_pairs.tsv"
    run_blocking(s1_te, s2_te, s3_te, "TEST", test_out)

    print("\n" + "=" * 60)
    print("  ✅ Phase 2 COMPLETE — proceed to Phase 3 (Feature Engineering)")
    print("=" * 60)


if __name__ == "__main__":
    main()
