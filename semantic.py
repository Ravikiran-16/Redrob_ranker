# Calculate similarity between candidates and the job description

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Job description used for comparison
JD_CORE_TEXT = """
Own the intelligence layer of Redrob's product: the ranking, retrieval, and
matching systems that decide what recruiters see when they search for
candidates and what candidates see when they search for roles. Audit
existing BM25 and rule-based scoring. Ship a v2 ranking system using
embeddings, hybrid retrieval, and LLM-based re-ranking. Set up evaluation
infrastructure: offline benchmarks, NDCG, MRR, MAP, online A/B testing,
recruiter-feedback loops. Production experience with embeddings-based
retrieval systems deployed to real users: sentence-transformers, OpenAI
embeddings, BGE, E5. Handled embedding drift, index refresh, retrieval
quality regression in production. Production experience with vector
databases or hybrid search infrastructure: Pinecone, Weaviate, Qdrant,
Milvus, OpenSearch, Elasticsearch, FAISS. Strong Python and code quality.
Hands-on experience designing evaluation frameworks for ranking systems.
Shipped at least one end-to-end ranking, search, or recommendation system
to real users at meaningful scale. Strong opinions about retrieval, hybrid
vs dense search, offline vs online evaluation, when to fine-tune vs prompt.
Applied ML and AI engineering at a product company, not a pure services or
academic research role. Built recommendation systems, search relevance,
information retrieval, ranking models, feature engineering for ML systems
before LLMs were fashionable, and is now extending that with modern LLM and
embedding techniques.
"""

# Calculate semantic score for each candidate
def build_semantic_scores(career_texts, jd_text=JD_CORE_TEXT, max_features=40000):

    # Create TF-IDF model
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        stop_words="english",
    )

    # Combine candidate profiles with job description
    corpus = career_texts + [jd_text]

    # Convert text into TF-IDF vectors
    tfidf = vectorizer.fit_transform(corpus)

    # Job description vector
    jd_vec = tfidf[-1]

    # Candidate vectors
    cand_matrix = tfidf[:-1]

    # Calculate similarity score
    sims = cosine_similarity(cand_matrix, jd_vec).ravel()

    # Normalize scores between 0 and 1
    if sims.max() > sims.min():
        sims = (sims - sims.min()) / (sims.max() - sims.min())

    # Return semantic scores
    return sims