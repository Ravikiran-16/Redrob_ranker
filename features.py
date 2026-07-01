"""
features.py
============
Pure-Python / regex feature extraction from a raw candidate record.

Design goal: every feature here is auditable -- you can point at the exact
field(s) of the candidate JSON that produced it. This is what lets the
`reasoning` column in the final CSV be built from real extracted facts
instead of hallucinated text, and what lets us defend the architecture in
the Stage 5 interview.

No network, no GPU, no heavy NLP model. Runs in O(text length) per candidate.
"""
import re
import datetime as dt
import jd_rubric as R

TODAY = dt.date(2026, 6, 27)


def _to_date(s):
    if not s:
        return None
    try:
        return dt.date.fromisoformat(s[:10])
    except Exception:
        return None


def _lower_join(*parts):
    return " ".join(p for p in parts if p).lower()


def _term_hits(text, terms):
    return [t for t in terms if t in text]


def extract_text_blob(cand):
    """All free text we treat as the candidate's 'narrative' for semantic matching."""
    p = cand.get("profile", {})
    parts = [
        p.get("headline", ""),
        p.get("summary", ""),
        p.get("current_title", ""),
    ]
    for ch in cand.get("career_history", []):
        parts.append(ch.get("title", ""))
        parts.append(ch.get("description", ""))
    # Skills contribute too, but with lower implicit weight than career text
    # (skills-list stuffing is the trap; career narrative is harder to fake
    # cheaply and is what we lean on for the "production AI" signal).
    skill_names = [s.get("name", "") for s in cand.get("skills", [])]
    parts.append(" ".join(skill_names))
    return _lower_join(*parts)


def extract_career_narrative(cand):
    """Career history + summary only, EXCLUDING the skills list. Used to detect
    real production AI/ML work vs. a stuffed skills array."""
    p = cand.get("profile", {})
    parts = [p.get("summary", ""), p.get("headline", "")]
    for ch in cand.get("career_history", []):
        parts.append(ch.get("title", ""))
        parts.append(ch.get("description", ""))
    return _lower_join(*parts)


def skill_lookup(cand):
    """name(lower) -> skill dict, for quick attribute lookups."""
    out = {}
    for s in cand.get("skills", []):
        out[s.get("name", "").strip().lower()] = s
    return out


def avg_tenure_months(career_history):
    if not career_history:
        return None
    durations = [ch.get("duration_months", 0) for ch in career_history]
    return sum(durations) / len(durations)


def detect_honeypot(cand):
    """Heuristic detector for 'subtly impossible' profiles.
    Returns (is_honeypot: bool, reasons: list[str])
    """
    reasons = []
    skills = cand.get("skills", [])
    yoe = cand.get("profile", {}).get("years_of_experience", 0) or 0

    # 1. "expert" proficiency with ~0 months of actual use
    expert_zero = [s["name"] for s in skills
                   if s.get("proficiency") == "expert" and s.get("duration_months", 0) <= 1]
    if len(expert_zero) >= 3:
        reasons.append(f"{len(expert_zero)} 'expert'-rated skills with ~0 months of use")

    # 2. Any single skill's duration_months exceeds total years of experience
    for s in skills:
        if s.get("duration_months", 0) > (yoe * 12 + 12):
            reasons.append(f"skill '{s.get('name')}' duration ({s.get('duration_months')}mo) exceeds total experience")
            break

    # 3. Career history total duration far exceeds stated years_of_experience
    ch = cand.get("career_history", [])
    total_months = sum(c.get("duration_months", 0) for c in ch)
    if yoe > 0 and total_months > (yoe * 12 * 1.4 + 12):
        reasons.append(f"career history totals {total_months}mo vs stated {yoe} years experience")

    # 4. Overlapping career_history date ranges (more than one "current" job, or
    #    start dates inconsistent with sequential careers) beyond a small tolerance
    starts = []
    for c in ch:
        sd = _to_date(c.get("start_date"))
        ed = _to_date(c.get("end_date")) or TODAY
        if sd:
            starts.append((sd, ed))
    starts.sort()
    overlap_months = 0
    for i in range(len(starts) - 1):
        end_i = starts[i][1]
        start_next = starts[i + 1][0]
        if end_i and start_next and end_i > start_next:
            overlap_months += (end_i - start_next).days / 30
    if overlap_months > 12:
        reasons.append(f"~{int(overlap_months)} months of overlapping concurrent roles")

    # 5. Education end_year before plausible start of career, or end_year in the future
    #    combined with years_of_experience implausibility is covered by #3 already.

    return (len(reasons) > 0, reasons)


def extract_features(cand):
    """Main entry point: returns a flat dict of engineered features for one candidate."""
    profile = cand.get("profile", {})
    signals = cand.get("redrob_signals", {})
    career = cand.get("career_history", [])
    skills = skill_lookup(cand)
    career_text = extract_career_narrative(cand)
    full_text = extract_text_blob(cand)

    feat = {}
    feat["candidate_id"] = cand["candidate_id"]
    feat["years_of_experience"] = profile.get("years_of_experience", 0) or 0
    feat["current_title"] = profile.get("current_title", "")
    feat["current_company"] = profile.get("current_company", "")
    feat["location"] = (profile.get("location") or "").lower()
    feat["country"] = (profile.get("country") or "").lower()
    feat["headline"] = profile.get("headline", "")
    feat["summary"] = profile.get("summary", "")

    # ---- Must-have capability detection (career narrative + skills) ----
    cap_scores = {}
    for cap, spec in R.MUST_HAVE_CAPABILITIES.items():
        hits = _term_hits(full_text, spec["skill_terms"])
        cap_scores[cap] = 1.0 if hits else 0.0
        feat[f"cap_{cap}_hit_terms"] = hits[:5]
    feat["cap_scores"] = cap_scores
    feat["must_have_coverage"] = sum(
        cap_scores[c] * R.MUST_HAVE_CAPABILITIES[c]["weight"] for c in cap_scores
    ) / sum(spec["weight"] for spec in R.MUST_HAVE_CAPABILITIES.values())

    # ---- Nice-to-have ----
    nice_hits = {}
    for cap, terms in R.NICE_TO_HAVE_CAPABILITIES.items():
        hits = _term_hits(full_text, terms)
        nice_hits[cap] = hits
    feat["nice_to_have_count"] = sum(1 for v in nice_hits.values() if v)
    feat["nice_to_have_hits"] = nice_hits

    # ---- Production AI evidence: must appear in CAREER NARRATIVE, not just skills ----
    prod_ai_hits = _term_hits(career_text, R.PRODUCTION_AI_SIGNAL_TERMS)
    feat["production_ai_evidence_count"] = len(prod_ai_hits)
    feat["production_ai_evidence_terms"] = prod_ai_hits[:6]

    # ---- Keyword-stuffing trap detector: many AI skill *names* listed but
    #      zero corroborating evidence in career narrative ----
    ai_skill_names_listed = [name for name in skills
                              if any(t in name for t in ["llm", "rag", "embedding", "gpt", "nlp",
                                                          "vector", "transformer", "fine-tun",
                                                          "lora", "ml", "ai "])]
    feat["ai_skill_names_listed_count"] = len(ai_skill_names_listed)
    feat["keyword_stuffing_flag"] = (
        len(ai_skill_names_listed) >= 5 and len(prod_ai_hits) == 0
    )

    # ---- Pre-LLM ML production experience (for the "LangChain wrapper" disqualifier) ----
    pre_llm_hits = _term_hits(career_text, R.PRE_LLM_ML_TERMS)
    feat["pre_llm_ml_evidence"] = len(pre_llm_hits) > 0
    langchain_only_hits = _term_hits(full_text, R.LANGCHAIN_OPENAI_WRAPPER_TERMS)
    # "recent (<12mo) projects using LangChain to call OpenAI" - approximate via:
    # langchain/openai wrapper terms present, AND no pre-LLM ML evidence, AND
    # most relevant experience is in current/most-recent role only.
    feat["langchain_wrapper_only_flag"] = bool(langchain_only_hits) and not pre_llm_hits and feat["production_ai_evidence_count"] <= 1

    # ---- Pure research disqualifier ----
    feat["pure_research_flag"] = any(t in career_text for t in R.PURE_RESEARCH_TERMS) and not any(
        ch.get("company", "").lower() not in ("",) and "university" not in ch.get("company", "").lower()
        and "institute" not in ch.get("company", "").lower()
        for ch in career  # crude proxy: has at least one non-academic employer
    ) is False  # placeholder, refined below
    # Simpler, robust version: research terms present AND ALL employers look academic/research
    research_term_present = any(t in career_text for t in R.PURE_RESEARCH_TERMS)
    non_academic_employer = any(
        not any(a in (ch.get("company", "") or "").lower() for a in ["university", "institute", "lab", "research"])
        for ch in career
    )
    feat["pure_research_flag"] = research_term_present and not non_academic_employer

    # ---- Consulting-only career ----
    employers = [ (ch.get("company") or "").strip().lower() for ch in career ]
    employers_all = employers + [feat["current_company"].lower()]
    is_consulting = lambda e: any(c in e for c in R.CONSULTING_ONLY_COMPANIES)
    feat["consulting_only_flag"] = len(employers_all) > 0 and all(is_consulting(e) for e in employers_all if e)

    # ---- CV/speech/robotics without NLP/IR exposure ----
    cv_hits = _term_hits(career_text, R.CV_SPEECH_ROBOTICS_TERMS)
    nlp_hits = _term_hits(career_text, R.NLP_IR_TERMS)
    feat["cv_only_flag"] = bool(cv_hits) and not nlp_hits

    # ---- Title chaser: short avg tenure across multiple companies ----
    tenure = avg_tenure_months(career)
    feat["avg_tenure_months"] = tenure
    feat["title_chaser_flag"] = (tenure is not None and tenure < R.TITLE_CHASER_MAX_AVG_TENURE_MONTHS
                                  and len(career) >= 3)

    # ---- Non-coding senior title drift ----
    feat["non_coding_title_flag"] = any(t in feat["current_title"].lower() for t in R.NON_CODING_TITLE_TERMS)

    # ---- Experience band fit (soft, triangular score peaking at ideal range) ----
    yoe = feat["years_of_experience"]
    lo, hi = R.EXPERIENCE_BAND
    ilo, ihi = R.IDEAL_EXPERIENCE
    if ilo <= yoe <= ihi:
        exp_fit = 1.0
    elif lo <= yoe < ilo:
        exp_fit = 0.7 + 0.3 * (yoe - lo) / max(ilo - lo, 1e-6)
    elif ihi < yoe <= hi:
        exp_fit = 1.0 - 0.3 * (yoe - ihi) / max(hi - ihi, 1e-6)
    elif yoe < lo:
        exp_fit = max(0.0, 0.5 - 0.15 * (lo - yoe))
    else:  # yoe > hi
        exp_fit = max(0.0, 0.7 - 0.07 * (yoe - hi))
    feat["experience_fit"] = max(0.0, min(1.0, exp_fit))

    # ---- Location fit ----
    # JD: "We don't sponsor work visas" outside India, but "open to relocation
    # candidates from Tier-1 Indian cities". So: India + HQ city = best;
    # India + other Tier-1 city + willing_to_relocate = good; India elsewhere
    # = ok-ish; outside India = heavily penalized regardless of skill match,
    # since the candidate cannot practically be hired without a visa we don't
    # sponsor -- unless they explicitly indicate relocation willingness AND
    # we have no visa signal to rely on (rare in this dataset, kept as a
    # narrow escape hatch rather than a hard block).
    loc = feat["location"]
    in_india = feat["country"] == R.COUNTRY_REQUIRED
    willing_relocate = bool(signals.get("willing_to_relocate", False))

    if any(h in loc for h in R.HQ_LOCATIONS):
        loc_fit = 1.0
    elif any(p in loc for p in R.PREFERRED_LOCATIONS):
        loc_fit = 0.85 if willing_relocate else 0.75
    elif in_india:
        loc_fit = 0.55 if willing_relocate else 0.40
    elif willing_relocate:
        loc_fit = 0.25   # outside India but says willing to relocate -- still no visa sponsorship, so capped low
    else:
        loc_fit = 0.05    # outside India, not relocating -- effectively unhireable for this role
    feat["location_fit"] = loc_fit

    # ---- Notice period fit ----
    notice = signals.get("notice_period_days")
    if notice is None:
        notice_fit = 0.6
    elif notice <= R.NOTICE_PERIOD_GOOD_DAYS:
        notice_fit = 1.0
    elif notice <= 60:
        notice_fit = 0.7
    else:
        notice_fit = 0.4
    feat["notice_fit"] = notice_fit
    feat["notice_period_days"] = notice

    # ---- Skill proficiency-weighted match against must-have skill terms ----
    matched_skill_quality = []
    for cap, spec in R.MUST_HAVE_CAPABILITIES.items():
        best = 0.0
        for name, sk in skills.items():
            if any(t in name for t in spec["skill_terms"]):
                prof_w = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.85, "expert": 1.0}.get(
                    sk.get("proficiency"), 0.5)
                dur = sk.get("duration_months", 0) or 0
                dur_w = min(1.0, dur / 24.0)  # ramps up to 2 years
                best = max(best, prof_w * (0.5 + 0.5 * dur_w))
        matched_skill_quality.append(best)
    feat["skill_quality_score"] = sum(matched_skill_quality) / len(matched_skill_quality) if matched_skill_quality else 0.0

    # ---- Behavioral / availability signals ----
    last_active = _to_date(signals.get("last_active_date"))
    days_inactive = (TODAY - last_active).days if last_active else 9999
    feat["days_inactive"] = days_inactive
    feat["recency_score"] = max(0.0, 1.0 - days_inactive / 180.0)  # ~0 after 6 months inactive

    feat["response_rate"] = signals.get("recruiter_response_rate", 0.0) or 0.0
    feat["open_to_work"] = bool(signals.get("open_to_work_flag", False))
    feat["interview_completion_rate"] = signals.get("interview_completion_rate", 0.5) or 0.0
    feat["profile_completeness"] = (signals.get("profile_completeness_score", 50) or 0) / 100.0
    feat["verified"] = bool(signals.get("verified_email")) and bool(signals.get("verified_phone"))
    feat["saved_by_recruiters_30d"] = signals.get("saved_by_recruiters_30d", 0) or 0
    feat["github_activity_score"] = signals.get("github_activity_score", -1)

    avail = (
        0.30 * feat["recency_score"]
        + 0.30 * feat["response_rate"]
        + 0.15 * (1.0 if feat["open_to_work"] else 0.4)
        + 0.15 * feat["interview_completion_rate"]
        + 0.10 * feat["profile_completeness"]
    )
    feat["availability_score"] = max(0.0, min(1.0, avail))

    # ---- Honeypot ----
    is_hp, hp_reasons = detect_honeypot(cand)
    feat["honeypot_flag"] = is_hp
    feat["honeypot_reasons"] = hp_reasons

    return feat
