import math
import pickle
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class InvertedIndex:
    """
    Inverted index over the pre-processed corpus.

    vocab:      term -> term_id
    postings:   term_id -> {pid: tf}  
    df:         term_id -> document frequency
    doc_norms:  pid -> Euclidean norm of the document's TF-IDF vector for cosine similarity
    n_docs:     number of indexed documents
    """
    vocab: Dict[str, int] = field(default_factory=dict)
    postings: Dict[int, Dict[str, int]] = field(default_factory=dict)
    df: Dict[int, int] = field(default_factory=dict)
    doc_norms: Dict[str, float] = field(default_factory=dict)
    n_docs: int = 0

    def idf(self, term_id: int) -> float:
        # Inverse document frequency: log(N / df).
        return math.log(self.n_docs / self.df[term_id])


def tf_weight(tf: int) -> float:
    # Term-frequency weight: 1 + log(tf).
    return 1.0 + math.log(tf)


def build_inverted_index(processed: pd.DataFrame, terms_col: str = "terms") -> InvertedIndex:
    """
    Build the inverted index from the processed corpus.

    :param processed: DataFrame with a 'pid' column and a list-of-terms column
    :param terms_col: column holding each document's list of stemmed terms
    :return: the InvertedIndex, including precomputed document norms
    """
    index = InvertedIndex()
    for pid, terms in zip(processed["pid"], processed[terms_col]):
        index.n_docs += 1
        for term, tf in Counter(terms).items():
            term_id = index.vocab.setdefault(term, len(index.vocab))
            index.postings.setdefault(term_id, {})[pid] = tf

    index.df = {tid: len(plist) for tid, plist in index.postings.items()}

    # Document vector norms: sqrt(sum_t (tf_weight * idf)^2)
    sq_sums: Dict[str, float] = {}
    for tid, plist in index.postings.items():
        idf = index.idf(tid)
        for pid, tf in plist.items():
            sq_sums[pid] = sq_sums.get(pid, 0.0) + (tf_weight(tf) * idf) ** 2
    index.doc_norms = {pid: math.sqrt(s) for pid, s in sq_sums.items()}
    return index


def save_index(index: InvertedIndex, path: str) -> None:
    """Persist the index with pickle."""
    with open(path, "wb") as f:
        pickle.dump(index, f)


def load_index(path: str) -> InvertedIndex:
    """Load an index saved with save_index."""
    with open(path, "rb") as f:
        return pickle.load(f)
