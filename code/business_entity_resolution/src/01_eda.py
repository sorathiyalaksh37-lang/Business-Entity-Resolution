"""
01_eda.py — Phase 1: Exploratory Data Analysis

Run from student_resource/ directory:
    python3 code/business_entity_resolution/src/01_eda.py

Outputs:
    - Printed EDA report to console
    - eda_report.txt saved to code/business_entity_resolution/
"""

import sys
import os
import re
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

# Make sure utils is importable
SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))
from utils import normalize_name, normalize_address, extract_pin_zip, extract_country_normalized

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parents[3]  # student_resource/
TRAIN_DIR = ROOT / "dataset" / "train"
TEST_DIR  = ROOT / "dataset" / "test"
OUT_DIR   = Path(__file__).parents[1]  # code/business_entity_resolution/

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    return df


def section(title: str, lines: list[str]) -> str:
    bar = "=" * 60
    return "\n" + bar + f"\n  {title}\n" + bar + "\n" + "\n".join(lines)


def describe_df(df: pd.DataFrame, name: str) -> list[str]:
    lines = []
    lines.append(f"File          : {name}")
    lines.append(f"Rows          : {len(df):,}")
    lines.append(f"Columns       : {list(df.columns)}")
    lines.append("")
    # Missing values
    for col in df.columns:
        n_empty = (df[col] == "").sum()
        pct = n_empty / len(df) * 100
        lines.append(f"  '{col}' empty: {n_empty:,} ({pct:.1f}%)")
    # Name length stats
    if "business_name" in df.columns:
        lengths = df["business_name"].str.len()
        lines.append(f"\n  business_name length — min:{lengths.min()} mean:{lengths.mean():.1f} max:{lengths.max()}")
    if "business_address" in df.columns:
        lengths = df["business_address"].str.len()
        lines.append(f"  business_address length — min:{lengths.min()} mean:{lengths.mean():.1f} max:{lengths.max()}")
    return lines


def country_dist(df: pd.DataFrame) -> list[str]:
    if "country" not in df.columns:
        return []
    counts = df["country"].value_counts(dropna=False)
    lines = []
    for country, n in counts.items():
        lines.append(f"  {str(country):<20} {n:>8,}  ({n/len(df)*100:.1f}%)")
    return lines


def analyze_ground_truth(gt: pd.DataFrame) -> list[str]:
    lines = []
    lines.append(f"Total S1 entities in ground truth : {len(gt):,}")

    # Parse matched_entity_ids
    def count_matches(s):
        s = str(s).strip()
        if s == "" or s == "nan":
            return 0
        return len([x for x in s.split(",") if x.strip()])

    gt["n_matches"] = gt["matched_entity_ids"].apply(count_matches)

    dist = Counter(gt["n_matches"].tolist())
    lines.append("\n  Match count distribution (how many S2/S3 matches per S1 entity):")
    for k in sorted(dist.keys()):
        pct = dist[k] / len(gt) * 100
        lines.append(f"    {k} matches: {dist[k]:,} entities ({pct:.1f}%)")

    # Source breakdown of matched IDs
    all_matched = []
    for row in gt["matched_entity_ids"]:
        row = str(row).strip()
        if row and row != "nan":
            all_matched.extend([x.strip() for x in row.split(",") if x.strip()])

    s2_count = sum(1 for x in all_matched if x.startswith("S2-"))
    s3_count = sum(1 for x in all_matched if x.startswith("S3-"))
    lines.append(f"\n  Total matched IDs: {len(all_matched):,}")
    lines.append(f"    From Source 2 (S2-): {s2_count:,}")
    lines.append(f"    From Source 3 (S3-): {s3_count:,}")
    return lines


def analyze_noise_patterns(df: pd.DataFrame, col: str, top_n: int = 20) -> list[str]:
    """Find most common tokens — helps identify abbreviation patterns."""
    lines = [f"  Top {top_n} tokens in '{col}':"]
    all_tokens = []
    for text in df[col].dropna():
        all_tokens.extend(str(text).lower().split())
    counter = Counter(all_tokens)
    for token, count in counter.most_common(top_n):
        lines.append(f"    '{token}': {count:,}")
    return lines


def analyze_abbrevs(df: pd.DataFrame) -> list[str]:
    """Check presence of known abbreviation variants in names."""
    lines = ["  Abbreviation variant analysis (business_name):"]
    abbrev_pairs = [
        ("pvt", "private"), ("ltd", "limited"), ("corp", "corporation"),
        ("inc", "incorporated"), ("co", "company"), ("intl", "international"),
        ("&", "and"), ("mfg", "manufacturing"),
    ]
    names = df["business_name"].str.lower()
    for abbr, full in abbrev_pairs:
        n_abbr = names.str.contains(r'\b' + re.escape(abbr) + r'\b', regex=True).sum()
        n_full = names.str.contains(r'\b' + re.escape(full) + r'\b', regex=True).sum()
        lines.append(f"    '{abbr}' appears {n_abbr:,}x  |  '{full}' appears {n_full:,}x")
    return lines


# ---------------------------------------------------------------------------
# Main EDA
# ---------------------------------------------------------------------------

def main():
    report_lines = []
    report_lines.append("BUSINESS ENTITY RESOLUTION — EDA REPORT")
    report_lines.append("Phase 1 Analysis")
    report_lines.append("")

    # ------------------------------------------------------------------
    # 1. Load training files
    # ------------------------------------------------------------------
    print("Loading training files...")
    s1_train = load_source(TRAIN_DIR / "train_source1.tsv")
    s2_train = load_source(TRAIN_DIR / "train_source2.tsv")
    s3_train = load_source(TRAIN_DIR / "train_source3.tsv")
    gt_train  = load_source(TRAIN_DIR / "train_ground_truth.tsv")

    print("Loading test files...")
    s1_test = load_source(TEST_DIR / "test_source1.tsv")
    s2_test = load_source(TEST_DIR / "test_source2.tsv")
    s3_test = load_source(TEST_DIR / "test_source3.tsv")

    # ------------------------------------------------------------------
    # 2. Basic stats
    # ------------------------------------------------------------------
    report_lines.append(section("SECTION 1: DATASET SIZES", [
        "--- TRAINING ---",
        *describe_df(s1_train, "train_source1.tsv"),
        "",
        *describe_df(s2_train, "train_source2.tsv"),
        "",
        *describe_df(s3_train, "train_source3.tsv"),
        "",
        "--- TEST ---",
        *describe_df(s1_test, "test_source1.tsv"),
        "",
        *describe_df(s2_test, "test_source2.tsv"),
        "",
        *describe_df(s3_test, "test_source3.tsv"),
    ]))

    # ------------------------------------------------------------------
    # 3. Country distribution
    # ------------------------------------------------------------------
    report_lines.append(section("SECTION 2: COUNTRY DISTRIBUTION", [
        "Source 1 TRAIN:", *country_dist(s1_train),
        "", "Source 2 TRAIN:", *country_dist(s2_train),
        "", "Source 3 TRAIN:", *country_dist(s3_train),
        "", "Source 1 TEST:", *country_dist(s1_test),
        "", "Source 2 TEST:", *country_dist(s2_test),
        "", "Source 3 TEST:", *country_dist(s3_test),
    ]))

    # ------------------------------------------------------------------
    # 4. Ground truth analysis
    # ------------------------------------------------------------------
    report_lines.append(section("SECTION 3: GROUND TRUTH ANALYSIS", analyze_ground_truth(gt_train)))

    # ------------------------------------------------------------------
    # 5. Noise pattern analysis
    # ------------------------------------------------------------------
    report_lines.append(section("SECTION 4: NAME ABBREVIATION ANALYSIS (Source 1 Train)", [
        *analyze_abbrevs(s1_train),
        "", *analyze_abbrevs(s2_train),
        "", *analyze_abbrevs(s3_train),
    ]))

    report_lines.append(section("SECTION 5: TOP NAME TOKENS (Source 1 Train)", [
        *analyze_noise_patterns(s1_train, "business_name", top_n=30),
    ]))

    report_lines.append(section("SECTION 6: TOP ADDRESS TOKENS (Source 1 Train)", [
        *analyze_noise_patterns(s1_train, "business_address", top_n=30),
    ]))

    # ------------------------------------------------------------------
    # 6. Normalization sanity check
    # ------------------------------------------------------------------
    print("Running normalization sanity check on sample records...")
    sample_rows = s1_train.sample(min(5, len(s1_train)), random_state=42)
    norm_lines = ["  Sample normalization output:"]
    for _, row in sample_rows.iterrows():
        raw_name = row.get("business_name", "")
        raw_addr = row.get("business_address", "")
        norm_lines.append(f"\n  RAW  name: {raw_name}")
        norm_lines.append(f"  NORM name: {normalize_name(raw_name)}")
        norm_lines.append(f"  RAW  addr: {raw_addr}")
        norm_lines.append(f"  NORM addr: {normalize_address(raw_addr)}")
        norm_lines.append(f"  PIN/ZIP  : {extract_pin_zip(raw_addr)}")
        norm_lines.append(f"  Country  : {extract_country_normalized(row.get('country', ''))}")
    report_lines.append(section("SECTION 7: NORMALIZATION SANITY CHECK", norm_lines))

    # ------------------------------------------------------------------
    # 7. Blocking complexity estimate
    # ------------------------------------------------------------------
    n_s1 = len(s1_test)
    n_s23 = len(s2_test) + len(s3_test)
    brute_force = n_s1 * n_s23
    report_lines.append(section("SECTION 8: BLOCKING COMPLEXITY ESTIMATE", [
        f"  Test S1 entities       : {n_s1:,}",
        f"  Test S2+S3 records     : {n_s23:,}",
        f"  Brute-force pairs      : {brute_force:,}",
        f"  Target candidates/S1   : ~100–500",
        f"  Target total candidates: ~{n_s1 * 200:,} (estimated)",
        f"  Reduction ratio target : {1 - (n_s1 * 200 / brute_force):.4f}",
    ]))

    # ------------------------------------------------------------------
    # 8. Print and save report
    # ------------------------------------------------------------------
    full_report = "\n".join(report_lines)
    print(full_report)

    report_path = OUT_DIR / "eda_report.txt"
    report_path.write_text(full_report, encoding="utf-8")
    print(f"\n✅ EDA report saved to: {report_path}")
    print("\n✅ Phase 1 Complete! Review the report, then proceed to Phase 2 (Blocking).")


if __name__ == "__main__":
    main()
