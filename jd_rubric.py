# Store job requirements and scoring rules
# These values are used while ranking candidates

# Job role details

ROLE_TITLE = "Senior AI Engineer - Founding Team"

EXPERIENCE_BAND = (5, 9)
IDEAL_EXPERIENCE = (6, 8)
IDEAL_APPLIED_AI_YEARS = (4, 5)

PREFERRED_LOCATIONS = {
    "pune", "noida", "delhi ncr", "delhi", "gurugram", "gurgaon",
    "mumbai", "hyderabad", "bangalore", "bengaluru"
}

HQ_LOCATIONS = {"pune", "noida"}

COUNTRY_REQUIRED = "india"

# Required skills for this job

MUST_HAVE_CAPABILITIES = {
    "embeddings_retrieval": {
        "skill_terms": [
            "sentence-transformers", "sentence transformers",
            "openai embeddings", "bge", "e5",
            "embeddings", "embedding",
            "dense retrieval", "semantic search",
            "retrieval", "rag", "vector search"
        ],
        "weight": 1.0,
    },

    "vector_db_hybrid_search": {
        "skill_terms": [
            "pinecone", "weaviate", "qdrant",
            "milvus", "opensearch",
            "elasticsearch", "faiss",
            "vector database",
            "hybrid search",
            "bm25"
        ],
        "weight": 1.0,
    },

    "python": {
        "skill_terms": ["python"],
        "weight": 0.6,
    },

    "eval_frameworks": {
        "skill_terms": [
            "ndcg",
            "mrr",
            "map",
            "a/b test",
            "ab testing",
            "offline evaluation",
            "evaluation framework",
            "learning to rank",
            "ranking metrics",
        ],
        "weight": 1.0,
    },
}

# Additional skills that are good to have

NICE_TO_HAVE_CAPABILITIES = {
    "llm_finetuning": [
        "lora",
        "qlora",
        "peft",
        "fine-tuning",
        "finetuning",
        "fine tuning",
    ],

    "ltr_models": [
        "xgboost",
        "lightgbm",
        "learning-to-rank",
        "neural ranking",
        "ranknet",
        "lambdamart",
    ],

    "hr_tech": [
        "hr-tech",
        "hr tech",
        "recruiting tech",
        "marketplace",
        "talent intelligence",
    ],

    "distributed_systems": [
        "distributed systems",
        "large-scale inference",
        "kubernetes",
        "spark",
        "kafka",
    ],

    "open_source": [
        "open source",
        "open-source",
        "github",
        "oss",
    ],
}

# Keywords to identify AI project experience

PRODUCTION_AI_SIGNAL_TERMS = [
    "embedding",
    "embeddings",
    "retrieval",
    "rag",
    "vector search",
    "vector database",
    "ranking system",
    "recommendation system",
    "recommender",
    "search relevance",
    "fine-tun",
    "llm",
    "nlp",
    "information retrieval",
    "semantic search",
    "ranking model",
    "re-ranking",
    "reranking",
    "ner",
    "transformer",
]

PRE_LLM_ML_TERMS = [
    "machine learning",
    "ml pipeline",
    "recommendation",
    "search ranking",
    "information retrieval",
    "nlp",
    "collaborative filtering",
    "click model",
    "ctr prediction",
    "feature engineering",
    "gradient boosting",
    "xgboost",
    "lightgbm",
    "logistic regression",
    "random forest",
]

LANGCHAIN_OPENAI_WRAPPER_TERMS = [
    "langchain",
    "openai api",
    "gpt-4",
    "gpt4",
    "chatgpt",
    "llamaindex",
]

# Conditions that reduce the candidate score

PURE_RESEARCH_TERMS = [
    "research scientist",
    "research associate",
    "postdoctoral",
    "phd researcher",
    "academic",
    "research lab",
    "research institute",
]

CONSULTING_ONLY_COMPANIES = {
    "tcs",
    "tata consultancy services",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "hcl",
    "tech mahindra",
    "mindtree",
    "ltimindtree",
}

CV_SPEECH_ROBOTICS_TERMS = [
    "computer vision",
    "speech recognition",
    "robotics",
    "autonomous driving",
    "image classification",
    "object detection",
    "speech-to-text",
    "tts",
    "lidar",
    "slam",
    "motion planning",
]

NLP_IR_TERMS = [
    "nlp",
    "natural language processing",
    "information retrieval",
    "search",
    "ranking",
    "retrieval",
    "recommendation",
    "embeddings",
    "text classification",
]

# Maximum average job duration

TITLE_CHASER_MAX_AVG_TENURE_MONTHS = 18

# Preferred notice period

NOTICE_PERIOD_GOOD_DAYS = 30
NOTICE_PERIOD_BUYOUT_DAYS = 30

# Senior non-coding job titles

NON_CODING_TITLE_TERMS = [
    "engineering manager",
    "director of engineering",
    "vp of engineering",
    "head of engineering",
    "principal architect",
    "enterprise architect",
]