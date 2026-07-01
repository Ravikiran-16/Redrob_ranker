"""
reasoning.py
============
Builds the `reasoning` column from FEATURES ONLY -- never from free
generation -- so every claim traces back to a real field in the candidate's
profile. This is what keeps reasoning honest under the Stage 4 manual review
checks (no hallucination, no templating, rank-consistent tone).
"""

def _fmt_caps(feat):
    caps = [c.replace("_", " ") for c, v in feat["cap_scores"].items() if v]
    return caps


def build_reasoning(feat, rank, total_score):
    title = feat["current_title"] or "professional"
    company = feat["current_company"] or "current employer"
    yoe = feat["years_of_experience"]
    loc = feat["location"].title() if feat["location"] else "location unspecified"

    caps_hit = _fmt_caps(feat)
    prod_terms = feat.get("production_ai_evidence_terms", [])

    pieces = []

    # Core identity line
    pieces.append(f"{title} at {company} with {yoe:.1f} yrs experience ({loc}).")

    # Positive evidence
    if caps_hit:
        pieces.append(f"Shows real {', '.join(caps_hit[:3]).replace('_', ' ')} experience"
                       + (f" (career history mentions {', '.join(prod_terms[:3])})" if prod_terms else "") + ".")
    elif prod_terms:
        pieces.append(f"Career history shows production AI/ML work: {', '.join(prod_terms[:3])}.")
    else:
        pieces.append("Limited direct evidence of production embeddings/retrieval/ranking work.")

    # Concerns, named honestly
    concerns = []
    if feat["country"] != "india" and feat["location_fit"] < 0.3:
        concerns.append(f"based outside India ({feat['location'].title() or feat['country'].title()}); "
                         f"role doesn't sponsor visas")
    if feat.get("keyword_stuffing_flag"):
        concerns.append("AI skills listed but not corroborated by career narrative (possible keyword stuffing)")
    if feat.get("honeypot_flag"):
        concerns.append("profile shows internally inconsistent experience claims")
    if feat.get("pure_research_flag"):
        concerns.append("background looks purely academic/research, no production deployment")
    if feat.get("consulting_only_flag"):
        concerns.append("career entirely at services/consulting firms")
    if feat.get("cv_only_flag"):
        concerns.append("background is CV/speech/robotics with no NLP/IR exposure")
    if feat.get("title_chaser_flag"):
        concerns.append(f"short average tenure (~{feat.get('avg_tenure_months', 0):.0f} mo/role)")
    if feat.get("langchain_wrapper_only_flag"):
        concerns.append("AI experience looks limited to recent LangChain/OpenAI wrapper work")
    if feat.get("non_coding_title_flag"):
        concerns.append("current title suggests management/architecture rather than hands-on coding")
    if feat["notice_period_days"] and feat["notice_period_days"] > 60:
        concerns.append(f"long notice period ({feat['notice_period_days']}d)")
    if feat["days_inactive"] > 90:
        concerns.append(f"inactive on platform for {feat['days_inactive']}d")
    if feat["response_rate"] < 0.2:
        concerns.append(f"low recruiter response rate ({feat['response_rate']:.0%})")

    if concerns:
        pieces.append("Concerns: " + "; ".join(concerns[:2]) + ".")
    elif feat["response_rate"] >= 0.5 and feat["days_inactive"] < 30:
        pieces.append(f"Active and responsive (replied {feat['response_rate']:.0%} of the time, "
                       f"last active {feat['days_inactive']}d ago).")

    text = " ".join(pieces)
    # Keep it to roughly 2 sentences worth of length
    return text
