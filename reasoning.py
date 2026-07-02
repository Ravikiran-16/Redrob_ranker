# Generate reasoning for each ranked candidate

def _fmt_caps(feat):
    # Get matched capabilities
    caps = [c.replace("_", " ") for c, v in feat["cap_scores"].items() if v]
    return caps


# Build explanation using extracted features
def build_reasoning(feat, rank, total_score):

    title = feat["current_title"] or "professional"
    company = feat["current_company"] or "current employer"
    yoe = feat["years_of_experience"]
    loc = feat["location"].title() if feat["location"] else "location unspecified"

    caps_hit = _fmt_caps(feat)
    prod_terms = feat.get("production_ai_evidence_terms", [])

    pieces = []

    # Basic candidate details
    pieces.append(f"{title} at {company} with {yoe:.1f} yrs experience ({loc}).")

    # Mention positive points
    if caps_hit:
        pieces.append(
            f"Shows real {', '.join(caps_hit[:3]).replace('_', ' ')} experience"
            + (f" (career history mentions {', '.join(prod_terms[:3])})"
               if prod_terms else "") + "."
        )

    elif prod_terms:
        pieces.append(
            f"Career history shows production AI/ML work: {', '.join(prod_terms[:3])}."
        )

    else:
        pieces.append(
            "Limited direct evidence of production embeddings/retrieval/ranking work."
        )

    concerns = []

    # Check for possible issues
    if feat["country"] != "india" and feat["location_fit"] < 0.3:
        concerns.append(
            f"based outside India ({feat['location'].title() or feat['country'].title()}); "
            f"role doesn't sponsor visas"
        )

    if feat.get("keyword_stuffing_flag"):
        concerns.append(
            "AI skills listed but not supported by career history"
        )

    if feat.get("honeypot_flag"):
        concerns.append(
            "profile has inconsistent experience details"
        )

    if feat.get("pure_research_flag"):
        concerns.append(
            "research background without production experience"
        )

    if feat.get("consulting_only_flag"):
        concerns.append(
            "career mainly in consulting companies"
        )

    if feat.get("cv_only_flag"):
        concerns.append(
            "computer vision background with limited NLP experience"
        )

    if feat.get("title_chaser_flag"):
        concerns.append(
            f"short average tenure (~{feat.get('avg_tenure_months',0):.0f} months)"
        )

    if feat.get("langchain_wrapper_only_flag"):
        concerns.append(
            "limited AI experience beyond LangChain/OpenAI"
        )

    if feat.get("non_coding_title_flag"):
        concerns.append(
            "current role is more management focused"
        )

    if feat["notice_period_days"] and feat["notice_period_days"] > 60:
        concerns.append(
            f"long notice period ({feat['notice_period_days']} days)"
        )

    if feat["days_inactive"] > 90:
        concerns.append(
            f"inactive for {feat['days_inactive']} days"
        )

    if feat["response_rate"] < 0.2:
        concerns.append(
            f"low recruiter response rate ({feat['response_rate']:.0%})"
        )

    # Add concerns if found
    if concerns:
        pieces.append("Concerns: " + "; ".join(concerns[:2]) + ".")

    # Otherwise mention positive activity
    elif feat["response_rate"] >= 0.5 and feat["days_inactive"] < 30:
        pieces.append(
            f"Active and responsive (replied {feat['response_rate']:.0%} of the time, "
            f"last active {feat['days_inactive']}d ago)."
        )

    text = " ".join(pieces)

    # Return final reasoning
    return text