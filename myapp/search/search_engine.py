import string
from pathlib import Path

import pandas as pd

from myapp.search.algorithms import (RANKINGS_LIST, closest_match_retrieve, conjunctive_retrieve, load_ratings,
                                     process_query, train_word2vec)
from myapp.search.indexing import build_inverted_index, load_index
from myapp.search.objects import ResultItem
from myapp.search.preprocessing import build_terms, compute_discount, preprocess_corpus

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class SearchEngine:
    """Class that implements the search engine logic"""

    def __init__(self, corpus):
        """
        Prepare the processed corpus and the inverted index once, when the application starts.
        The files saved by Parts 1 and 2 are loaded if both exist; otherwise both are built in memory.

        :param corpus: the documents corpus (pid -> Document)
        """
        processed_path = DATA_DIR / "processed_corpus.pkl"
        index_path = DATA_DIR / "inverted_index.pkl"
        if processed_path.exists() and index_path.exists():
            self.processed = pd.read_pickle(processed_path)
            self.index = load_index(index_path)
        else:
            self.processed = preprocess_corpus(corpus)
            self.index = build_inverted_index(self.processed)
        # needed by the custom ranking
        load_ratings(self.processed)
        self.word2vec_ready = False

    def search(self, search_query, search_id, corpus, ranking="bm25", top_k=20):
        """
        Search the products that contain all the query terms and rank them.
        If no product contains all of them, return the ones that match the most terms.

        :param search_query: the text written by the user
        :param search_id: id of this search, added to the result links
        :param corpus: the documents corpus (pid -> Document)
        :param ranking: "tfidf", "bm25", "custom" or "word2vec; bm25 by default, because we got the best results in Part 3"
        :param top_k: maximum number of results (None = all)
        :return: (results, info). results is a list of ResultItem, best first. info is a dictionary with
                 total (matches before top_k), partial (True if no product has all the terms),
                 matched and words (terms matched and distinct query terms) and ignored (unknown words)
        """
        print("Search query:", search_query)

        ranker = RANKINGS_LIST.get(ranking)
        if ranker is None:
            raise ValueError(f"Unknown ranking function: {ranking}. Available: {sorted(RANKINGS_LIST)}")

        # word2vec is trained only the first time it is requested
        if ranking == "word2vec" and not self.word2vec_ready:
            train_word2vec(self.processed)
            self.word2vec_ready = True

        terms = process_query(search_query)
        words = len(set(terms))
        candidates = conjunctive_retrieve(terms, self.index)
        matched = words if candidates else 0
        partial = False
        if not candidates:
            terms, candidates, matched = closest_match_retrieve(terms, self.index)
            partial = bool(candidates)

        ranked = ranker(terms, self.index, candidates)
        info = {"total": len(ranked), "partial": partial, "matched": matched, "words": words,
                "ignored": self._unknown_words(search_query)}

        results = []
        for pid, score in ranked[:top_k]:
            doc = corpus[pid]
            data = doc.model_dump()
            data["discount"] = compute_discount(doc)
            results.append(ResultItem(**data,
                                      details_url="doc_details?pid={}&search_id={}".format(pid, search_id),
                                      ranking=score))
        return results, info

    def _unknown_words(self, search_query):
        """Words of the query, as the user wrote them, that do not appear in the catalogue."""
        unknown = []
        for word in search_query.split():
            terms = build_terms(word)  # empty for stop words, which are not reported
            if terms and not any(term in self.index.vocab for term in terms):
                unknown.append(word.strip(string.punctuation))
        return list(dict.fromkeys(unknown))
