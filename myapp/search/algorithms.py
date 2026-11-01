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


def search_in_corpus(query: str, index: InvertedIndex, top_k: int = None, ranking: str = "tfidf") -> List[Tuple[str, float]]:
    """
    Full pipeline: process query -> AND retrieval -> ranking.

    :param query: raw query text
    :param index: the InvertedIndex
    :param top_k: keep only the best k results (None = all)
    :param ranking: ranking function name; currently only "tfidf"
    :return: [(pid, score)] sorted by descending score
    """
    terms = process_query(query)
    candidates = conjunctive_retrieve(terms, index)
    if ranking == "tfidf":
        ranked = rank_tfidf(terms, index, candidates)
    else:
        raise ValueError(f"Unknown ranking function: {ranking}")
    return ranked[:top_k] if top_k else ranked
