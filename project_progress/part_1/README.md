# Part 1 — Text Processing and Exploratory Data Analysis

## Content

| File | Description |
|------|-------------|
| `IRWA_part_1.ipynb` | Code: corpus loading, pre-processing pipeline, handling of each field, EDA |

The pre-processing code is not in the notebook but in `myapp/search/preprocessing.py`, so Parts 2–4 reuse exactly the
same functions (`build_terms`, `preprocess_corpus`, ...). The notebook imports it and explains it.

Output of this part: `data/processed_corpus.pkl`, one row per document with the original fields, the processed text
fields (`title_terms`, `description_terms`, `details_terms`, `brand_terms`, `sub_category_terms`), the merged `terms`
list and the typed numerical attributes. Part 2 loads it to build the inverted index.

## How to reproduce the results

1. Put the data files in the `data` folder of the project root:
   - `data/fashion_products_dataset.json`
   - `data/validation_labels.csv`
2. From the project root, create and activate a virtual environment and install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open `project_progress/part_1/IRWA_part_1.ipynb` and run all cells from top to bottom. The notebook locates the
   project root automatically (first parent folder containing `myapp`), so it can be launched from any directory.
   The first run downloads the NLTK resources (stop words, punkt).