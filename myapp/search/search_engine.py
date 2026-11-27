from pathlib import Path

import pandas as pd

from myapp.search.algorithms import load_ratings, search_in_corpus, train_word2vec
from myapp.search.indexing import build_inverted_index, load_index
from myapp.search.objects import ResultItem
from myapp.search.preprocessing import preprocess_corpus

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

        :param search_query: the text written by the user
        :param search_id: id of this search, added to the result links
        :param corpus: the documents corpus (pid -> Document)
        :param ranking: "tfidf", "bm25", "custom" or "word2vec; bm25 by default, because we got the best results in Part 3"
        :param top_k: maximum number of results (None = all)
        :return: list of ResultItem, best first
        """
        print("Search query:", search_query)

        # word2vec is trained only the first time it is requested
        if ranking == "word2vec" and not self.word2vec_ready:
            train_word2vec(self.processed)
            self.word2vec_ready = True

        results = []
        for pid, score in search_in_corpus(search_query, self.index, top_k=top_k, ranking=ranking):
            doc = corpus[pid]
            results.append(ResultItem(pid=pid, title=doc.title, description=doc.description,
                                      url="doc_details?pid={}&search_id={}".format(pid, search_id),
                                      ranking=score))
        return results
