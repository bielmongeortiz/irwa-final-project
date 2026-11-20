import math
from collections import Counter
from typing import List, Set, Tuple

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


# Registry of ranking functions. Every ranker has the signature
# (query_terms, index, candidates) -> [(pid, score)] sorted by descending score.
RANKINGS_LIST = {"tfidf": rank_tfidf, "bm25": rank_bm25}


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
