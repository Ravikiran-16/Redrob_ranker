import json
import sys
import os

# Add parent folder so we can import rank.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from rank import run_pipeline

# Configure Streamlit page
st.set_page_config(page_title="Redrob Ranker Sandbox", layout="wide")

# Page title
st.title("Redrob Candidate Ranker — Sandbox")

# Short description for users
st.caption(
    "Upload a small JSONL sample (one candidate JSON object per line, ≤100 candidates) "
    "to verify the ranking pipeline runs end-to-end. The full 100K-candidate run is the "
    "`rank.py` CLI script in this repo; this sandbox is the lightweight reproducibility check."
)

# Upload candidate file
uploaded = st.file_uploader(
    "candidates sample (.jsonl)",
    type=["jsonl", "json", "txt"]
)

# Path of the default sample dataset
sample_path = os.path.join(
    os.path.dirname(__file__),
    "sample_candidates_mini.jsonl"
)

# Option to use sample data if no file is uploaded
use_bundled = st.checkbox(
    "Use the bundled sample instead of uploading",
    value=uploaded is None
)

records = []

# Read candidate data from bundled sample
if use_bundled and os.path.exists(sample_path):

    with open(sample_path) as f:
        records = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    st.info(f"Using bundled sample: {len(records)} candidates.")

# Read uploaded candidate file
elif uploaded is not None:

    text = uploaded.read().decode("utf-8")

    records = [
        json.loads(line)
        for line in text.splitlines()
        if line.strip()
    ]

    st.info(f"Loaded {len(records)} candidates from upload.")

# Continue only if candidate data is available
if records:

    # Sandbox supports only first 100 candidates
    if len(records) > 100:

        st.warning(
            f"Sample has {len(records)} candidates; truncating to first 100."
        )

        records = records[:100]

    # Run ranking when button is clicked
    if st.button("Run ranker", type="primary"):

        log_lines = []

        # Execute ranking pipeline
        with st.spinner("Running pipeline..."):

            results = run_pipeline(
                records,
                top_n=min(100, len(records)),
                log=lambda m: log_lines.append(m)
            )

        st.success(f"Ranked {len(records)} candidates.")

        # Show pipeline logs
        with st.expander("Pipeline log"):

            for l in log_lines:
                st.text(l)

        rows = []

        # Prepare output for display
        for rank, score, feat, reasoning_text in results:

            rows.append({
                "rank": rank,
                "score": score,
                "candidate_id": feat["candidate_id"],
                "current_title": feat["current_title"],
                "reasoning": reasoning_text,
            })

        # Display ranked candidates
        st.dataframe(
            rows,
            use_container_width=True
        )

        # Convert results into CSV
        csv_text = (
            "candidate_id,rank,score,reasoning\n"
            + "\n".join(
                f'{r["candidate_id"]},{r["rank"]},{r["score"]:.4f},"{r["reasoning"].replace(chr(34), chr(39))}"'
                for r in rows
            )
        )

        # Download ranked results
        st.download_button(
            "Download ranked CSV",
            csv_text,
            file_name="sandbox_ranking.csv"
        )

# Show message if no dataset is available
else:

    st.info(
        "Upload a .jsonl sample or check 'Use bundled sample' to get started."
    )