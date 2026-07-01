"""
sandbox/app.py
================
Minimal Streamlit sandbox satisfying submission_spec.md Section 10.5:
accepts a small candidate sample (<=100 candidates) and runs the same
ranking pipeline end-to-end in-browser, producing a ranked table.

Run locally:
    streamlit run sandbox/app.py

Deploy: push this repo to GitHub, then create a Streamlit Cloud app
pointing at sandbox/app.py (or an HF Space using the streamlit SDK).
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from rank import run_pipeline

st.set_page_config(page_title="Redrob Ranker Sandbox", layout="wide")
st.title("Redrob Candidate Ranker — Sandbox")
st.caption(
    "Upload a small JSONL sample (one candidate JSON object per line, ≤100 candidates) "
    "to verify the ranking pipeline runs end-to-end. The full 100K-candidate run is the "
    "`rank.py` CLI script in this repo; this sandbox is the lightweight reproducibility check."
)

uploaded = st.file_uploader("candidates sample (.jsonl)", type=["jsonl", "json", "txt"])

sample_path = os.path.join(os.path.dirname(__file__), "sample_candidates_mini.jsonl")
use_bundled = st.checkbox("Use the bundled sample instead of uploading", value=uploaded is None)

records = []
if use_bundled and os.path.exists(sample_path):
    with open(sample_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    st.info(f"Using bundled sample: {len(records)} candidates.")
elif uploaded is not None:
    text = uploaded.read().decode("utf-8")
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    st.info(f"Loaded {len(records)} candidates from upload.")

if records:
    if len(records) > 100:
        st.warning(f"Sample has {len(records)} candidates; truncating to first 100 for the sandbox check.")
        records = records[:100]

    if st.button("Run ranker", type="primary"):
        log_lines = []
        with st.spinner("Running pipeline..."):
            results = run_pipeline(records, top_n=min(100, len(records)),
                                    log=lambda m: log_lines.append(m))

        st.success(f"Ranked {len(records)} candidates.")
        with st.expander("Pipeline log"):
            for l in log_lines:
                st.text(l)

        rows = []
        for rank, score, feat, reasoning_text in results:
            rows.append({
                "rank": rank,
                "score": score,
                "candidate_id": feat["candidate_id"],
                "current_title": feat["current_title"],
                "reasoning": reasoning_text,
            })
        st.dataframe(rows, use_container_width=True)

        csv_text = "candidate_id,rank,score,reasoning\n" + "\n".join(
            f'{r["candidate_id"]},{r["rank"]},{r["score"]:.4f},"{r["reasoning"].replace(chr(34), chr(39))}"'
            for r in rows
        )
        st.download_button("Download ranked CSV", csv_text, file_name="sandbox_ranking.csv")
else:
    st.info("Upload a .jsonl sample or check 'use bundled sample' to get started.")
