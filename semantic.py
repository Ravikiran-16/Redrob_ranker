"""
semantic.py
============
Lightweight semantic relevance layer.

Why TF-IDF and not a transformer embedding model?
- Compute constraints are hard: CPU-only, <=5 min wall-clock, no network
  during ranking, <=16GB RAM, for 100,000 candidates.
- A from-scratch TF-IDF + cosine-similarity over the full corpus is exact,
  deterministic, dependency-light (sklearn only), and comfortably fits the
  budget (fit+transform on 100k short documents is seconds, not minutes).
- Pretrained sentence embeddings (e.g. sentence-transformers) would add
  real semantic quality, but require ~80-400MB of model weights to be
  bundled as a precomputed artifact and a non-trivial CPU pass per
  candidate. We support this as an OPTIONAL upgrade path (see
  precompute_embeddings.py) but the ranker must work correctly with TF-IDF
  alone, which is what ships by default.

Why TF-IDF still beats naive keyword counting:
- It downweights generic words shared by every profile (years, experience,
  engineer...) and upweights words that are discriminative for this JD's
  vocabulary (retrieval, embeddings, ranking, evaluation...).
- We deliberately build the vector from CAREER NARRATIVE TEXT, not the raw
  skills list, so that a candidate who has stuffed 10 AI skill *names* into
  their skills array without ever describing AI work in a role gets a LOW
  semantic score here -- exactly the keyword-stuffing trap the JD warns about.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# The JD's own "core intent" text -- the paragraphs that actually describe
# the work and the ideal candidate, intentionally excluding boilerplate
# (comp/logistics, hackathon meta-notes) so the vector space isn't diluted.
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


def build_semantic_scores(career_texts, jd_text=JD_CORE_TEXT, max_features=40000):
    """
    career_texts: list[str] -- one career-narrative document per candidate,
                  in the SAME order the candidates will be written out.
    Returns: np.ndarray of cosine similarities in [0, 1], same length/order.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        stop_words="english",
    )
    corpus = career_texts + [jd_text]
    tfidf = vectorizer.fit_transform(corpus)
    jd_vec = tfidf[-1]
    cand_matrix = tfidf[:-1]
    sims = cosine_similarity(cand_matrix, jd_vec).ravel()
    # Min-max normalize to spread the distribution across [0,1] for easier blending
    if sims.max() > sims.min():
        sims = (sims - sims.min()) / (sims.max() - sims.min())
    return sims
