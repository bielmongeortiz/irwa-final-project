import math
from collections import Counter
from typing import List, Set, Tuple

import numpy as np
from gensim.models import Word2Vec

from myapp.search.indexing import InvertedIndex, tf_weight
from myapp.search.preprocessing import build_terms


def process_query(query: str) -> List[str]:
    #Apply the same pre-processing pipeline used for documents to the query text.
    return build_terms(query)


def conjunctive_retrieve(query_terms: List[str], index: InvertedIndex) -> Set[str]:
    """
    AND semantics: return the pids of documents containing every query term.
    A term outside the vocabulary makes the result empty.
    Postings are intersected starting from the rarest term.
    """
    if not query_terms:
        return set()
    term_ids = []
    for term in set(query_terms):
        tid = index.vocab.get(term)
        if tid is None:
            return set()
        term_ids.append(tid)
    term_ids.sort(key=lambda t: index.df[t])
    result = set(index.postings[term_ids[0]])
    for tid in term_ids[1:]:
        result &= index.postings[tid].keys()
        if not result:
            break
    return result


def rank_tfidf(query_terms: List[str], index: InvertedIndex, candidates: Set[str]) -> List[Tuple[str, float]]:
    """
    Rank candidate documents by cosine similarity between TF-IDF vectors.
    Weights: (1 + log tf) * idf for both query and documents.

    :return: [(pid, score)] sorted by descending score (ties broken by pid)
    """
    q_tf = Counter(query_terms)
    q_weights = {}
    for term, tf in q_tf.items():
        tid = index.vocab.get(term)
        if tid is not None:
            q_weights[tid] = tf_weight(tf) * index.idf(tid)
    q_norm = math.sqrt(sum(w * w for w in q_weights.values()))
    if q_norm == 0:
        return []

    scores = {pid: 0.0 for pid in candidates}
    for tid, qw in q_weights.items():
        idf = index.idf(tid)
        plist = index.postings[tid]
        for pid in candidates:
            tf = plist.get(pid)
            if tf:
                scores[pid] += qw * tf_weight(tf) * idf
    ranked = [(pid, s / (q_norm * index.doc_norms[pid])) for pid, s in scores.items()]
    ranked.sort(key=lambda x: (-x[1], x[0]))
    return ranked


LENGTHS_BY_INDEX = {}


def document_lengths(index: InvertedIndex) -> Tuple[dict, float]:
    """
    Length of each document (number of indexed terms, counting repetitions) and the average length.
    Computed once per index from the postings and cached, so the index file does not change.
    """
    key = id(index)
    if key not in LENGTHS_BY_INDEX:
        lengths = Counter()
        for plist in index.postings.values():
            for pid, tf in plist.items():
                lengths[pid] += tf
        avg = sum(lengths.values()) / len(lengths) if lengths else 0.0
        LENGTHS_BY_INDEX[key] = (dict(lengths), avg)
    return LENGTHS_BY_INDEX[key]


def rank_bm25(query_terms: List[str], index: InvertedIndex, candidates: Set[str],
              k1: float = 1.5, b: float = 0.75) -> List[Tuple[str, float]]:
    """
    Rank candidate documents with Okapi BM25.
    idf(t) = log(N / df), as in the lecture definition.
    score(q, d) = sum over the distinct query terms of idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))

    :param k1: term frequency saturation
    :param b: length normalisation (0 = none, 1 = full)
    :return: [(pid, score)] sorted by descending score (ties broken by pid)
    """
    lengths, avgdl = document_lengths(index)
    scores = {pid: 0.0 for pid in candidates}
    for term in set(query_terms):
        tid = index.vocab.get(term)
        if tid is None:
            continue
        df = index.df[tid]
        idf = math.log(index.n_docs / df)
        plist = index.postings[tid]
        for pid in candidates:
            tf = plist.get(pid)
            if tf:
                scores[pid] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths[pid] / avgdl))
    ranked = list(scores.items())
    ranked.sort(key=lambda x: (-x[1], x[0]))
    return ranked


# Average rating of every product (pid -> rating), used by rank_custom.
# It is filled once with load_ratings(processed_corpus).
RATINGS = {}


def load_ratings(processed) -> None:
    """
    Store the average rating of every product for the custom score.
    A product without rating gets the mean rating of the corpus, so the rating does not move it up or down.
    """
    mean_rating = processed["average_rating"].mean()
    RATINGS.clear()
    for pid, rating in zip(processed["pid"], processed["average_rating"]):
        if math.isnan(rating):
            rating = mean_rating
        RATINGS[pid] = rating


def rank_custom(query_terms: List[str], index: InvertedIndex, candidates: Set[str],
                alpha: float = 0.9) -> List[Tuple[str, float]]:
    """
    Our own score: BM25 combined with the product rating.
    score(q, d) = alpha * bm25(q, d) / max_bm25 + (1 - alpha) * rating(d) / 5
    max_bm25 is the best BM25 score among the candidates, so both parts are between 0 and 1.
    With alpha = 0.9 the text decides; the rating only changes the order of documents with almost the same BM25.

    :return: [(pid, score)] sorted by descending score (ties broken by pid)
    """
    if not RATINGS:
        raise ValueError("Call load_ratings(processed_corpus) before using the custom ranking.")

    bm25_ranked = rank_bm25(query_terms, index, candidates)
    if not bm25_ranked:
        return []
    max_bm25 = bm25_ranked[0][1]

    ranked = []
    for pid, bm25_score in bm25_ranked:
        score = alpha * bm25_score / max_bm25 + (1 - alpha) * RATINGS[pid] / 5
        ranked.append((pid, score))
    ranked.sort(key=lambda x: (-x[1], x[0]))
    return ranked


# Word2vec vectors, used by rank_word2vec. They are filled once with train_word2vec(processed_corpus).
WORD_VECTORS = {}   # term -> vector of the term
DOC_VECTORS = {}    # pid -> vector of the document (average of its term vectors)


def text_vector(terms: List[str]):
    """
    Vector of a text: the average of the word2vec vectors of its terms.
    Terms without a vector are skipped. Returns None if no term has a vector.
    """
    vectors = []
    for term in terms:
        if term in WORD_VECTORS:
            vectors.append(WORD_VECTORS[term])
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


def train_word2vec(processed, vector_size: int = 100, window: int = 5) -> Word2Vec:
    """
    Train word2vec on the terms of the corpus and compute the vector of every document.
    workers=1 and seed=42 make the training give the same vectors every time.

    :param processed: the processed corpus DataFrame (columns pid and terms)
    :param vector_size: number of dimensions of the vectors
    :param window: how many words at each side count as the context of a word
    :return: the trained model
    """
    model = Word2Vec(sentences=list(processed["terms"]), vector_size=vector_size, window=window,
                     min_count=1, workers=1, seed=42)

    WORD_VECTORS.clear()
    for word in model.wv.index_to_key:
        WORD_VECTORS[word] = model.wv[word]

    DOC_VECTORS.clear()
    for pid, terms in zip(processed["pid"], processed["terms"]):
        vector = text_vector(terms)
        if vector is not None:
            DOC_VECTORS[pid] = vector
    return model


def cosine_similarity(a, b) -> float:
    """Cosine of the angle between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def rank_word2vec(query_terms: List[str], index: InvertedIndex, candidates: Set[str]) -> List[Tuple[str, float]]:
    """
    Rank candidate documents by the cosine similarity between the word2vec vector of the query
    and the word2vec vector of each document. (index is not used, it is kept so all rankers have the same signature.)

    :return: [(pid, score)] sorted by descending score (ties broken by pid)
    """
    if not DOC_VECTORS:
        raise ValueError("Call train_word2vec(processed_corpus) before using the word2vec ranking.")

    query_vector = text_vector(query_terms)
    if query_vector is None:
        return []

    ranked = []
    for pid in candidates:
        ranked.append((pid, cosine_similarity(query_vector, DOC_VECTORS[pid])))
    ranked.sort(key=lambda x: (-x[1], x[0]))
    return ranked


# Registry of ranking functions. Every ranker has the signature
# (query_terms, index, candidates) -> [(pid, score)] sorted by descending score.
RANKINGS_LIST = {"tfidf": rank_tfidf, "bm25": rank_bm25, "custom": rank_custom, "word2vec": rank_word2vec}


def search_in_corpus(query: str, index: InvertedIndex, top_k: int = None, ranking: str = "tfidf") -> List[Tuple[str, float]]:
    """
    Full pipeline: process query -> AND retrieval -> ranking.

    :param query: raw query text
    :param index: the InvertedIndex
    :param top_k: keep only the best k results (None = all)
    :param ranking: a key of RANKINGS_LIST
    :return: [(pid, score)] sorted by descending score
    """
    ranker = RANKINGS_LIST.get(ranking)
    if ranker is None:
        raise ValueError(f"Unknown ranking function: {ranking}. Available: {sorted(RANKINGS_LIST)}")
    terms = process_query(query)
    candidates = conjunctive_retrieve(terms, index)
    ranked = ranker(terms, index, candidates)
    return ranked if top_k is None else ranked[:top_k]
