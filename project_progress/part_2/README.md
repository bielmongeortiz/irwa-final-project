# Part 2 — Indexing and Evaluation

## Content

| File | Description |
|------|-------------|
| `IRWA_part_2.ipynb` | Code and results: inverted index, test queries, TF-IDF ranking, evaluation metrics and their results |
| `test_queries_labels.csv` | Top 20 results of our five test queries, with our relevance labels in the `label` column |
| `../../myapp/search/indexing.py` | Inverted index: build, save and load |
| `../../myapp/search/algorithms.py` | Query processing, conjunctive (AND) retrieval and TF-IDF ranking |
| `../../myapp/search/evaluation.py` | Evaluation metrics |

The code is not in the notebook but in `myapp/search/`, so that the later parts and the web application can import the same functions. The notebook imports it and shows its results.

Output of this part: `data/inverted_index.pkl`, the saved inverted index. The `data` folder is not in the repository, so the file is created when the notebook runs.

## Requirements

- Python 3 (the whole notebook was run with Python 3.11).
- The packages in `requirements.txt`.
- The data file `data/validation_labels.csv`.
- The file `data/processed_corpus.pkl`, created by Part 1. See [the Part 1 README](../part_1/README.md).

## How to run it, step by step

1. Put `fashion_products_dataset.json` and `validation_labels.csv` in the `data` folder of the project root.
2. From the project root, create and activate a virtual environment and install the dependencies:
   ```bash
   python -m venv irwa_venv
   source irwa_venv/bin/activate      # Windows: irwa_venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Register the virtual environment as a Jupyter kernel. Otherwise Jupyter may run the notebook with another Python installation and fail with `No module named 'nltk'`:
   ```bash
   python -m ipykernel install --user --name irwa_venv
   ```
4. If `data/processed_corpus.pkl` does not exist, create it by running the Part 1 notebook or the code in [the Part 1 README](../part_1/README.md).
5. Start Jupyter, open the notebook and select the `irwa_venv` kernel:
   ```bash
   jupyter notebook project_progress/part_2/IRWA_part_2.ipynb
   ```
6. Run all cells from top to bottom. The notebook finds the project root by itself, so it can be started from any folder.

To run the whole notebook without opening it (from the project root):

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=irwa_venv project_progress/part_2/IRWA_part_2.ipynb
```

The index must have 28,080 documents and a vocabulary of 5,053 terms.

## Relevance labels of the test queries

`test_queries_labels.csv` holds our judgments, so running the notebook does not need any manual step. If the file already exists, the notebook does not overwrite it.

To judge the queries again:

1. Delete the file (this loses the current labels) and run the notebook, which writes it again with the `label` column empty.
2. Fill the `label` column with 1 (relevant) or 0 (not relevant) and keep commas as the separator.
3. Run the notebook again. Section 7.1 computes the metrics only when all the labels are filled in.

## Functions and options

### Search (`myapp/search/algorithms.py`)

| Function | What it does |
|----------|--------------|
| `process_query(query)` | Applies to the query the same pre-processing as to the documents |
| `conjunctive_retrieve(query_terms, index)` | Returns the documents that contain all the query terms |
| `rank_tfidf(query_terms, index, candidates)` | Ranks the candidates by cosine similarity between TF-IDF vectors |
| `search_in_corpus(query, index, top_k=None, ranking="tfidf")` | Full pipeline: process query, AND retrieval and ranking |

From the project root:

```python
import pandas as pd
from myapp.search.indexing import build_inverted_index
from myapp.search.algorithms import search_in_corpus

processed = pd.read_pickle("data/processed_corpus.pkl")
index = build_inverted_index(processed)

search_in_corpus("men slim jeans blue", index, top_k=3)   # the best 3 results, as (pid, score)
search_in_corpus("men slim jeans blue", index)            # all the 217 matching documents
```

**Options.** `top_k` limits the number of results. `ranking` selects the ranking function; in this part the only one is `"tfidf"`.

### Index (`myapp/search/indexing.py`)

| Function | What it does |
|----------|--------------|
| `build_inverted_index(processed, terms_col="terms")` | Builds the index from the processed corpus. `terms_col` selects the column of terms to index |
| `save_index(index, path)` / `load_index(path)` | Saves and loads the index with pickle |

### Evaluation metrics (`myapp/search/evaluation.py`)

All the metrics receive `labels`, the relevance labels of the results in ranking order. `n_relevant` is the number of relevant documents of the query, including the ones that were not retrieved.

| Function | Metric |
|----------|--------|
| `precision_at_k(labels, k)` | Precision@K |
| `recall_at_k(labels, k, n_relevant)` | Recall@K |
| `f1_at_k(labels, k, n_relevant)` | F1-Score@K |
| `average_precision(labels, n_relevant)` | Average Precision of the whole ranking |
| `average_precision_at_k(labels, k, n_relevant)` | Average Precision@K |
| `mean_average_precision(ap_per_query)` | Mean Average Precision (MAP) |
| `reciprocal_rank(labels)` | Reciprocal Rank of one query |
| `mean_reciprocal_rank(rr_per_query)` | Mean Reciprocal Rank (MRR) |
| `dcg_at_k(labels, k)` | Discounted Cumulative Gain@K |
| `ndcg_at_k(labels, k, judged_labels)` | Normalized DCG@K (NDCG). `judged_labels` holds the labels of all the judged documents of the query |
| `labels_in_rank_order(ranked_pids, judgments, drop_unjudged=True)` | Turns a ranking of pids into its list of labels |
| `count_relevant(labels)` | Number of relevant documents in a list of labels |

```python
from myapp.search.evaluation import precision_at_k, recall_at_k

labels = [1, 0, 1, 0, 1]
precision_at_k(labels, 3)      # 0.667
recall_at_k(labels, 3, 4)      # 0.5
```

**Options.**
- `drop_unjudged=True` skips the documents without a label. `drop_unjudged=False` counts them as non-relevant.
- The cutoffs K (5, 10 and 20) are set in the list `[5, 10, 20]` of the function `per_query_table` in the notebook.

## Notebook sections

| Section | What it reproduces |
|---------|--------------------|
| 1 | Loading of the processed corpus |
| 2 | Building and saving the inverted index, with checks on a small example and on the corpus |
| 3 | Term popularity and choice of the five test queries |
| 4 | Conjunctive retrieval and TF-IDF ranking of the test queries |
| 5 | Evaluation metrics, checked with the worked examples of the course slides |
| 6 | Validation queries: how much of each ranking is judged and which relevant documents are not retrieved |
| 6.1 | Metrics of the validation queries using only the judged documents |
| 6.2 | The same metrics counting the unjudged documents as non-relevant |
| 7 | Creation of `test_queries_labels.csv` |
| 7.1 | Metrics of our test queries |
| 8 | Figures used as evidence in the limitations section of the report |
