"""
jd_rubric.py
============
This module is the "deep job understanding" layer.

Rather than re-parsing the JD text at ranking time (which would force the
ranker to either call an LLM per-candidate -- forbidden by the compute
constraints -- or rely on fragile keyword matching), we do the hard semantic
reading of the JD *once*, by hand/at design time, and encode what it actually
means as a structured rubric. This is the standard pattern for production
recruiting systems: a human (or an LLM call made ONCE on the JD, not per
candidate) turns a messy JD into a structured spec; the per-candidate scoring
step is then cheap, deterministic and auditable.

Every field below traces back to a specific paragraph in job_description.docx.
"""

# ---------------------------------------------------------------------------
# Role framing
# ---------------------------------------------------------------------------
ROLE_TITLE = "Senior AI Engineer - Founding Team"

EXPERIENCE_BAND = (5, 9)          # soft band, not a hard cutoff (JD: "a range, not a requirement")
IDEAL_EXPERIENCE = (6, 8)         # "ideal candidate ... 6-8 years total experience"
IDEAL_APPLIED_AI_YEARS = (4, 5)   # "of which 4-5 are in applied ML/AI roles at product companies"

PREFERRED_LOCATIONS = {"pune", "noida", "delhi ncr", "delhi", "gurugram", "gurgaon",
                        "mumbai", "hyderabad", "bangalore", "bengaluru"}
HQ_LOCATIONS = {"pune", "noida"}
COUNTRY_REQUIRED = "india"   # visas not sponsored outside India -> heavy penalty, not hard block

# ---------------------------------------------------------------------------
# Must-have technical capabilities ("Things you absolutely need")
# Each capability is expressed as a family of keywords/skill names so we can
# detect it however the candidate phrased it (no single magic keyword).
# Production-deployment evidence is required, not just a skills-list mention.
# ---------------------------------------------------------------------------
MUST_HAVE_CAPABILITIES = {
    "embeddings_retrieval": {
        "skill_terms": ["sentence-transformers", "sentence transformers", "openai embeddings",
                         "bge", "e5", "embeddings", "embedding", "dense retrieval", "semantic search",
                         "retrieval", "rag", "vector search"],
        "weight": 1.0,
    },
    "vector_db_hybrid_search": {
        "skill_terms": ["pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
                         "faiss", "vector database", "hybrid search", "bm25"],
        "weight": 1.0,
    },
    "python": {
        "skill_terms": ["python"],
        "weight": 0.6,
    },
    "eval_frameworks": {
        "skill_terms": ["ndcg", "mrr", "map", "a/b test", "ab testing", "offline evaluation",
                         "evaluation framework", "learning to rank", "ranking metrics"],
        "weight": 1.0,
    },
}

# "Things we'd like you to have but won't reject you for"
NICE_TO_HAVE_CAPABILITIES = {
    "llm_finetuning": ["lora", "qlora", "peft", "fine-tuning", "finetuning", "fine tuning"],
    "ltr_models": ["xgboost", "lightgbm", "learning-to-rank", "neural ranking", "ranknet", "lambdamart"],
    "hr_tech": ["hr-tech", "hr tech", "recruiting tech", "marketplace", "talent intelligence"],
    "distributed_systems": ["distributed systems", "large-scale inference", "kubernetes", "spark", "kafka"],
    "open_source": ["open source", "open-source", "github", "oss"],
}

# Core production-AI signal terms used to separate "actually built AI systems"
# from "lists AI words in the skills section". This list is intentionally
# broad so that career-history text (not just the skills array) is searched.
PRODUCTION_AI_SIGNAL_TERMS = [
    "embedding", "embeddings", "retrieval", "rag", "vector search", "vector database",
    "ranking system", "recommendation system", "recommender", "search relevance",
    "fine-tun", "llm", "nlp", "information retrieval", "semantic search",
    "ranking model", "re-ranking", "reranking", "ner", "transformer",
]

PRE_LLM_ML_TERMS = [
    "machine learning", "ml pipeline", "recommendation", "search ranking",
    "information retrieval", "nlp", "collaborative filtering", "click model",
    "ctr prediction", "feature engineering", "gradient boosting", "xgboost",
    "lightgbm", "logistic regression", "random forest",
]

LANGCHAIN_OPENAI_WRAPPER_TERMS = ["langchain", "openai api", "gpt-4", "gpt4", "chatgpt", "llamaindex"]

# ---------------------------------------------------------------------------
# Disqualifiers / strong negative signals ("we will not / probably not move forward")
# These are *soft* in our system: they apply a large multiplicative penalty
# rather than a hard removal, because the JD itself says "we'll seriously
# consider candidates outside the band if other signals are strong" and lists
# these as "probably not" rather than absolute rules (except pure-research).
# ---------------------------------------------------------------------------
PURE_RESEARCH_TERMS = ["research scientist", "research associate", "postdoctoral", "phd researcher",
                        "academic", "research lab", "research institute"]

CONSULTING_ONLY_COMPANIES = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mindtree", "ltimindtree",
}

CV_SPEECH_ROBOTICS_TERMS = ["computer vision", "speech recognition", "robotics", "autonomous driving",
                             "image classification", "object detection", "speech-to-text", "tts",
                             "lidar", "slam", "motion planning"]
NLP_IR_TERMS = ["nlp", "natural language processing", "information retrieval", "search", "ranking",
                "retrieval", "recommendation", "embeddings", "text classification"]

TITLE_CHASER_MAX_AVG_TENURE_MONTHS = 18   # "switching every 1.5 years"

# Notice period preference
NOTICE_PERIOD_GOOD_DAYS = 30
NOTICE_PERIOD_BUYOUT_DAYS = 30

# Things we explicitly do NOT want -> title patterns that signal architecture/TL
# drift away from writing code (only penalized if also senior/long tenure in that role)
NON_CODING_TITLE_TERMS = ["engineering manager", "director of engineering", "vp of engineering",
                           "head of engineering", "principal architect", "enterprise architect"]
