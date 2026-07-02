#!/usr/bin/env python3
# Main file to rank candidates
# Reads candidate data, calculates scores, and creates the final CSV
import argparse
import csv
import gzip
import json
import sys
import time

import numpy as np

import features as F
import jd_rubric as R
import semantic as S
import reasoning as RZ

# Open candidate file
def open_candidates(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")

# Calculate final score for one candidate
def compute_final_score(feat, semantic_score):
    base_fit = (
        0.38 * semantic_score
        + 0.24 * feat["must_have_coverage"]
        + 0.14 * feat["skill_quality_score"]
        + 0.09 * feat["experience_fit"]
        + 0.15 * feat["location_fit"]
    )

    penalty = 1.0
    if feat["pure_research_flag"]:
        penalty *= 0.05
    if feat["consulting_only_flag"]:
        penalty *= 0.30
    if feat["cv_only_flag"]:
        penalty *= 0.30
    if feat["langchain_wrapper_only_flag"]:
        penalty *= 0.40
    if feat["title_chaser_flag"]:
        penalty *= 0.70
    if feat["non_coding_title_flag"]:
        penalty *= 0.70
    if feat["keyword_stuffing_flag"]:
        penalty *= 0.40
    if feat["honeypot_flag"]:
        penalty *= 0.01

    availability_mult = 0.5 + 0.5 * feat["availability_score"]
    notice_mult = 0.7 + 0.3 * feat["notice_fit"]

    score = base_fit * penalty * availability_mult * notice_mult
    return max(0.0, min(1.0, score))


def run_pipeline(candidate_records, top_n=100, log=lambda msg: None):
    # Run the complete ranking process
    import time as _time
    # Start timer
    t0 = _time.time()
    # Store extracted features
    feats = []
    career_texts = []
    # Extract features from each candidate
    for cand in candidate_records:
        feat = F.extract_features(cand)
        feats.append(feat)
        career_texts.append(F.extract_career_narrative(cand))

    n = len(feats)
    t1 = _time.time()
    log(f"loaded + feature-extracted {n} candidates in {t1 - t0:.1f}s")
# Calculate semantic scores
    semantic_scores = S.build_semantic_scores(career_texts)
    t2 = _time.time()
    log(f"semantic scoring done in {t2 - t1:.1f}s")

    scored = []
    for feat, sem in zip(feats, semantic_scores):
        final_score = compute_final_score(feat, float(sem))
        rounded_score = round(final_score, 4)
        scored.append((rounded_score, feat))
# Sort candidates by score
    scored.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))
    top = scored[:top_n]
    t3 = _time.time()
    log(f"scored + sorted {n} candidates in {t3 - t2:.1f}s")
# Store final ranked candidates
    results = []
    for rank, (score, feat) in enumerate(top, start=1):
        reasoning_text = RZ.build_reasoning(feat, rank, score)# Generate explanation
        results.append((rank, score, feat, reasoning_text))
    return results


def main():
    ap = argparse.ArgumentParser()# Read input arguments
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-n", type=int, default=100)
    args = ap.parse_args()

    t0 = time.time()
# Read input arguments
    candidate_records = []
    with open_candidates(args.candidates) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidate_records.append(json.loads(line))

    n = len(candidate_records)
    print(f"[rank.py] read {n} candidate records", file=sys.stderr)
# Run ranking pipeline
    results = run_pipeline(candidate_records, top_n=args.top_n,
                            log=lambda msg: print(f"[rank.py] {msg}", file=sys.stderr))
# Count suspicious profiles
    honeypots_in_top = sum(1 for _, _, feat, _ in results if feat["honeypot_flag"])
    print(f"[rank.py] honeypots in top {args.top_n}: {honeypots_in_top} "
          f"({100.0 * honeypots_in_top / args.top_n:.1f}%)", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, score, feat, reasoning_text in results:
            writer.writerow([feat["candidate_id"], rank, f"{score:.4f}", reasoning_text])

    t4 = time.time()
    print(f"[rank.py] wrote {args.out}. Total wall time: {t4 - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":# Start the program
    main()
