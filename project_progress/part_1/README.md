# Part 1 — Text Processing and Exploratory Data Analysis

## Content

| File | Description |
|------|-------------|
| `IRWA_part_1.ipynb` | Code and results: corpus loading, pre-processing pipeline, handling of each field, EDA |
| `../../myapp/search/preprocessing.py` | The pre-processing code itself |

The pre-processing code is not in the notebook but in `myapp/search/preprocessing.py`, so that the later indexing and web application can import the same functions. The notebook imports it and shows its results.

Output of this part: `data/processed_corpus.pkl`, one row per document with the original fields, the processed text
fields (`title_terms`, `description_terms`, `details_terms`, `brand_terms`, `sub_category_terms`), the merged `terms`
list and the typed numerical attributes. 

## Requirements

- Python 3.13 (`.python-version`; `pyproject.toml` accepts >=3.9).
- The packages in `requirements.txt` (pandas, nltk, pydantic, jupyter, matplotlib, seaborn, ...).
- The NLTK English stop-word list, downloaded automatically the first time `preprocessing.py` is imported.
- The data files `data/fashion_products_dataset.json` and `data/validation_labels.csv`.

## How to run it, step by step

1. Put the two data files in the `data` folder of the project root.
2. From the project root, create and activate a virtual environment and install the dependencies:
   ```bash
   python -m venv irwa_venv
   source irwa_venv/bin/activate
   pip install -r requirements.txt
   ```
3. Start Jupyter and open the notebook:
   ```bash
   jupyter notebook project_progress/part_1/IRWA_part_1.ipynb
   ```
4. Run all cells from top to bottom. The notebook locates the project root automatically (first parent folder
   containing `myapp`), so it can be launched from any directory.
5. Result: `data/processed_corpus.pkl` with 28,080 rows and 21 columns.

To run only the pre-processing, without the notebook (from the project root):

```python
import pandas as pd
from myapp.search.load_corpus import load_corpus
from myapp.search.preprocessing import preprocess_corpus

corpus = load_corpus("data/fashion_products_dataset.json")   
processed = preprocess_corpus(corpus)                        
processed.to_pickle("data/processed_corpus.pkl")
processed = pd.read_pickle("data/processed_corpus.pkl")      
```

## Functions and options (`myapp/search/preprocessing.py`)

| Function | What it does |
|----------|--------------|
| `normalize_text(text)` | NFKC normalization, lowercasing, domain normalizations and punctuation removal |
| `tokenize(text)` | Splits on whitespace; drops 1-character tokens and pure numbers |
| `build_terms(text, remove_stop_words=True, stemming=True)` | Full pipeline for one text: normalize → tokenize → stop words → stemming. Used for documents **AND** queries |
| `details_to_text(product_details)` | Keeps only the descriptive values of `product_details` as text |
| `clean_seller(seller)` | Removes the scraping artifact from seller names |
| `compute_discount(doc)` | Discount in %; derived from the prices when missing |
| `preprocess_document(doc)` | Pre-processed record of one document (raw fields, term lists, numerical attributes) |
| `preprocess_corpus(corpus)` | Applies `preprocess_document` to the whole corpus and returns a dataframe indexed by `pid` |

**Options.** `build_terms` can switch off each step independently:

```python
from myapp.search.preprocessing import build_terms

build_terms("Slim jeans with pockets")                           # ['slim', 'jean', 'pocket']
build_terms("Slim jeans with pockets", stemming=False)           # ['slim', 'jeans', 'pockets']
build_terms("Slim jeans with pockets", remove_stop_words=False)  # ['slim', 'jean', 'with', 'pocket']
```

**Settings you can edit** (module constants):
- `NORMALIZATIONS`: the domain normalization rules (regex, replacement).
- `DETAIL_KEYS` and `BOOLEAN_DETAIL_KEYS`: which `product_details` keys are turned into text.
- `STOP_WORDS`: the stop-word set (NLTK English by default).

**Which fields are indexed.** `terms` is the concatenation of `title_terms + description_terms + details_terms +
brand_terms + sub_category_terms`. To index other fields, change the `record["terms"]` line of `preprocess_document`.

## Notebook sections

| Section | What it reproduces |
|---------|--------------------|
| 1.1 | Loading of the corpus (`raw` and `corpus`) |
| 1.2 | Step-by-step example of the pipeline, query examples and the effect of each step on the vocabulary |
| 1.3 | Pre-processing of the whole corpus and creation of `processed_corpus.pkl` |
| 1.4 | Profile of the text fields and which fields to index (validation queries) |
| 1.5 | Numerical fields: missing values, discount consistency, summary statistics |
| 2.1 | EDA: overview, categories and missing values |
| 2.2 | EDA: word counts, sentence length and vocabulary (`figures/word_counts.png`) |
| 2.3 | EDA: most frequent terms and word cloud (`figures/word_cloud.png`) |
| 2.4 | EDA: price, discount and rating (`figures/numerical_distributions.png`) |
| 2.5 | EDA: top brands and sellers, out-of-stock products |
| 2.6 | EDA: data quality checks (duplicated titles, gender in title vs URL, seller vs stock) |
