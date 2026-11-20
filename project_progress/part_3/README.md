# Part 3 — Ranking and Filtering

## Content

| File | Description |
|------|-------------|
| `IRWA_part_3.ipynb` | Code and results: ranking TF-IDF + cosine similarity, ranking BM25, ranking our own score, ranking word2vec + cosine similarity |
| `../../myapp/search/algorithms.py` | differents rankings used |

The code is not in the notebook but in `myapp/search/`, so that the later parts and the web application can import the same functions. The notebook imports it and shows its results.

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
   jupyter notebook project_progress/part_3/IRWA_part_3.ipynb
   ```
6. Run all cells from top to bottom. The notebook finds the project root by itself, so it can be started from any folder.

To run the whole notebook without opening it (from the project root):

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=irwa_venv project_progress/part_3/IRWA_part_3.ipynb
```

## Functions

### Search (`myapp/search/algorithms.py`)

| Function | What it does |
|----------|--------------|
| `document_lengths(index)` | Length of each document and average length |
| `rank_bm25(query_terms, index, candidates)` | Ranks the candidates with Okapi BM25 |
| `load_ratings(processed)` | Store average rating of every product for the custom score |
| `rank_custom(query_terms, index, candidates,alpha)` | Ranks the candidates with our own Score |
| `text_vector(terms)` | Returns the average of the word2vec vectors of its terms |
| `train_word2vec(processed, vector_size, window)` | Train word2vec on the terms of the corpus and compute the vector of every document |
| `rank_word2vec(query_terms, index, candidates)` | Rank candidate documents by the cosine similarity between the word2vec vector of the query and the word2vec vector of each document |



## Notebook sections

| Section | What it reproduces |
|---------|--------------------|
| 1 | Load the index and the corpus |
| 2 | Test queries |
| 3 | Ranking TF-IDF + cosine similarity |
| 3.1 | Evaluation |
| 4 | Ranking BM25 |
| 4.1 | Evaluation |
| 5 | Ranking our own Score |
| 5.1 | Evaluation |
| 6 | Ranking word2vec + cosine similarity |
| 6.1 | Evaluation |