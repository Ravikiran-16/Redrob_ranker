# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

A CPU-only, no-network, sub-2-minute ranker for the 100,000-candidate pool,
built around one core idea: **separate "understanding the JD" (done once,
by a human, encoded as a rubric) from "scoring a candidate" (done 100,000
times, cheaply and deterministically)**. This is the only way to satisfy
the compute budget (≤5 min, ≤16 GB RAM, CPU-only, no network) without
collapsing into keyword matching.

## Quick start

```bash
pip install -r requirements.txt

## Running the ranker

[1. Place candidates.jsonl in this folder (or use your own path)
2. Run:
   python rank.py --candidates candidates.jsonl --out submission.csv]
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

python validate_submission.py ./submission.csv   # provided by organizers
```



Measured on a 1-vCPU / 4GB container (worse than the 16GB/CPU sandbox the
organizers will use): **~60-90 seconds wall-clock, ~2GB peak RAM** for the
full 100,000-candidate pool. Comfortably inside the 5-minute budget with no
GPU and no network calls.

## Architecture

```
candidates.jsonl ──► features.py ──► per-candidate engineered features
                                         │
JD (read once,    ──► jd_rubric.py ─────┤  (must-haves, disqualifiers,
hand-encoded)                           │   ideal profile, location rules)
                                         ▼
career narrative text ──► semantic.py ──► TF-IDF cosine similarity to JD
 (career_history +                          core-intent text
  summary, NOT skills list)
                                         │
                                         ▼
                              rank.py: blend into final_score
                                         │
                                         ▼
                          reasoning.py: fact-grounded explanation
                                         │
                                         ▼
                                  submission.csv (top 100)
```

### 1. Deep job understanding (`jd_rubric.py`)

The JD is long, narrative, and explicitly warns against shallow keyword
matching ("the right answer is not candidates whose skills section contains
the most AI keywords"). Re-deriving that understanding per-candidate would
require an LLM call per candidate — explicitly forbidden by the compute
constraints. Instead we did the semantic reading **once**, at design time,
and encoded it as a structured rubric:

- Must-have capability families (embeddings/retrieval, vector DB/hybrid
  search, Python, eval frameworks for ranking) as keyword *families*, not
  single magic strings.
- Soft experience band (5-9y, ideal 6-8y) — triangular fit score, not a
  hard cutoff, matching the JD's own framing ("a range, not a requirement").
- Explicit disqualifier patterns: pure-research-only careers, services/
  consulting-only careers (TCS/Infosys/Wipro/etc.), CV/speech/robotics
  without NLP/IR exposure, "LangChain-wrapper-only" AI experience, title-
  chasers (short average tenure), senior titles that have drifted away from
  hands-on coding.
- Location/visa reality: HQ cities > other Tier-1 Indian cities > rest of
  India > outside India (no sponsorship, heavily penalized regardless of
  skill match, with a narrow allowance for explicit relocation intent).

This is the layer a reviewer should interrogate first in the Stage 5
interview — it's where the actual judgment calls live, not buried inside
ML hyperparameters.

### 2. Signal extraction (`features.py`)

Pure Python/regex, one pass per candidate, every feature traceable to a
specific JSON field:

- **Capability + skill-quality scoring**: matches must-have capability
  families against the candidate's full text, then separately computes a
  proficiency × duration-weighted score for the actual matched skill
  entries (an "expert" skill used for 0 months scores low; "intermediate"
  used for 3 years scores respectably).
- **Keyword-stuffing trap detector**: counts AI-sounding skill *names* in
  the skills array vs. AI evidence terms found in the **career narrative**
  (titles + role descriptions), separately. A candidate with 8 AI skill
  names and zero corroborating career text is flagged — this is exactly the
  trap the JD describes ("AI keywords listed as skills but title is
  'Marketing Manager'").
- **Honeypot detector**: flags "expert" proficiency with ~0 months of use,
  any skill duration exceeding total stated experience, career-history
  total duration far exceeding stated years of experience, and overlapping
  concurrent roles beyond a small tolerance. These map directly to the
  "subtly impossible profile" honeypots described in `redrob_signals_doc`.
  Honeypots aren't hard-removed; they're score-multiplied by 0.01, which in
  practice sinks them off the list without needing to special-case them.
- **Behavioral availability score**: blends recency of last activity,
  recruiter response rate, open-to-work flag, interview completion rate,
  and profile completeness into a 0-1 multiplier on the base fit score —
  per the JD's explicit instruction to down-weight perfect-on-paper but
  unavailable candidates.
- **Disqualifier flags**: pure-research, consulting-only, CV/speech/
  robotics-only, LangChain-wrapper-only, title-chaser, non-coding-title —
  each becomes a multiplicative penalty rather than a hard filter, because
  the JD itself hedges most of these ("probably not", "we'll seriously
  consider ... if other signals are strong").

### 3. Contextual relevance, not keyword count (`semantic.py`)

A TF-IDF vector space (1-2 grams, sublinear TF, English stopwords removed)
is fit once across all 100,000 candidates' **career narratives** (titles +
role descriptions + summary — explicitly *not* the skills array) plus the
JD's core-intent paragraph. Cosine similarity against the JD vector gives a
contextual relevance score that:

- Naturally downweights generic words every profile shares ("years",
  "experience", "engineer") via IDF.
  upweights words discriminative for this JD's vocabulary (retrieval,
  embeddings, ranking, evaluation, hybrid search).
- Rewards a candidate who *describes* building a recommendation system at a
  product company even if they never use the word "RAG" — the Tier-5 case
  the JD explicitly calls out.
- Cannot be gamed by stuffing the skills array, because skills aren't in
  the vector for this score (they're scored separately, deliberately, with
  lower weight, in `skill_quality_score`).

We chose TF-IDF over a pretrained sentence-embedding model
(sentence-transformers/BGE/E5) for this submission specifically because of
the compute constraints: TF-IDF fit+transform over 100k short documents
takes ~20-35s with zero model-weight downloads, no network, no GPU, and
fully deterministic output. `semantic.py`'s docstring documents the
sentence-embedding upgrade path if the compute budget were relaxed.

### 4. Score blending (`rank.py`)

```
base_fit = 0.38*semantic + 0.24*must_have_coverage + 0.14*skill_quality
           + 0.09*experience_fit + 0.15*location_fit

penalty  = product of disqualifier multipliers (0.01 - 1.0, see jd_rubric.py)

final    = base_fit * penalty * availability_multiplier * notice_multiplier
```

Behavioral signals (availability, notice period) are applied as
*multipliers*, not additive terms — a candidate with a perfect skills/
semantic match but a 5% response rate and 6 months of inactivity should
fall well below a slightly-weaker-but-actually-reachable candidate. This
mirrors the JD's own instruction almost verbatim.

Ties are broken by `candidate_id` ascending, and scores are rounded to 4
decimals *before* sorting so the tie-break the validator checks for is
internally consistent.

### 5. Honest, grounded reasoning (`reasoning.py`)

The `reasoning` column is built entirely from the same feature dict used
for scoring — never freely generated. Every sentence is templated around
*actual extracted values* (current title/company/years, which capability
keywords were actually found in career text, which concern flags fired).
This is what keeps it safe under the Stage 4 manual-review checks: no
hallucinated skills, no rank/tone mismatch (concerns are only surfaced when
they actually exist for that candidate, so low-rank entries read with
appropriate hedging and high-rank entries don't).

## What's intentionally *not* here

- No per-candidate LLM calls (forbidden by spec, and wouldn't fit the
  5-minute budget for 100k candidates regardless).
- No GPU dependency.
- No network calls during ranking — `semantic.py`'s TF-IDF is fit from the
  local corpus, not downloaded.

## Known limitations / honest tradeoffs

- TF-IDF is a bag-of-n-grams; it doesn't have true paraphrase understanding
  the way a transformer embedding would (e.g. "built search that finds
  similar items" vs "recommendation system" share fewer n-grams than a
  dense embedding would place close together). The capability-family
  keyword matching in `features.py` is the deliberate backstop for this —
  it casts a wide net of phrasings per capability rather than relying on
  exact terms.
- The disqualifier penalties are hand-tuned multipliers, not learned from
  labeled data (there's no ground truth available at submission time to
  learn against). They're designed to be directionally correct and
  defensible rather than precisely calibrated.
- Honeypot detection is heuristic (internal-consistency checks), not
  exhaustive — it will not catch every possible "subtly impossible"
  construction, but it doesn't need to: it needs to keep the honeypot rate
  in the top 100 under 10%, which it does (0% on this run).

## Reproducing the submission CSV

```bash
python rank.py --candidates ./candidates.jsonl --out ./team_xxx.csv
```

No pre-computation step is required — there is no embedding index or model
checkpoint to build ahead of time. The entire pipeline (load → feature
extraction → TF-IDF fit/transform → score → write CSV) runs inside the
5-minute ranking-step budget.

## Sandbox / small-sample demo

See `sandbox/app.py` — a minimal Streamlit app that accepts a small JSONL
sample (≤100 candidates) and runs the same `rank.py` pipeline end-to-end,
producing a ranked table in-browser. Deploy to Streamlit Cloud / HF Spaces
by pointing at this repo with `sandbox/app.py` as the entrypoint.
