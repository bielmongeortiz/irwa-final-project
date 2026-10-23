import re
import string
import unicodedata
from typing import Dict, List, Optional

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from myapp.search.objects import Document

nltk.download("stopwords", quiet=True)

STEMMER = PorterStemmer()
STOP_WORDS = set(stopwords.words("english")) 

# Domain normalizations applied before tokenization
NORMALIZATIONS = [
    (re.compile(r"\bt[\s\-]?shirts?\b"), " tshirt "),                    # t-shirt / t shirt / tshirts -> tshirt
    (re.compile(r"\b(?:women['’]?s|woman['’]?s|woman|ladies|lady)\b"), " women "),
    (re.compile(r"\b(?:men['’]?s|man['’]?s)\b"), " men "),
    (re.compile(r"\bpoly\s?cotton\b"), " poly cotton "),                  # polycotton -> poly + cotton
    (re.compile(r"\bslim\s?fit\b"), " slim fit "),                        # slimfit -> slim + fit
    (re.compile(r"https?://\S+|www\.\S+"), " "),                         # urls
]

PUNCTUATION_TABLE = str.maketrans({c: " " for c in string.punctuation + "’‘“”–—•"})

# product_details keys that describe the product as a user would search for it.
DETAIL_KEYS = ["Fabric", "Pattern", "Color", "Secondary Color", "Sleeve", "Type", "Ideal For", "Fit",
               "Generic Name", "Neck Type", "Neck", "Collar", "Occasion", "Closure", "Rise", "Length Type",
               "Sole Material"]
# Yes/No keys: the key name itself becomes a term when the value is "Yes".
BOOLEAN_DETAIL_KEYS = ["Hooded", "Reversible", "Faded", "Distressed"]


SELLER_ARTIFACT = re.compile(r"\d(?:\.\d)?Seller changed.*$")

# Fields shown in the search results; copied as they come from Document
OUTPUT_FIELDS =["pid", "title", "description", "brand", "category", "sub_category", "product_details",
                 "seller", "out_of_stock", "selling_price", "discount", "actual_price", "average_rating", "url"]


def normalize_text(text: Optional[str]) -> str:
    if not text:  # None or empty string (many descriptions are empty)
        return ""
    # turns non-breaking spaces (\xa0) into normal spaces so split() works
    text = unicodedata.normalize("NFKC", text).lower()
    # normalizations before punctuation removal
    for pattern, replacement in NORMALIZATIONS:
        text = pattern.sub(replacement, text)
    return text.translate(PUNCTUATION_TABLE)


def tokenize(text: str) -> List[str]:
    # whitespace split 
    return [t for t in text.split() if len(t) > 1 and not t.isdigit()]


def build_terms(text: Optional[str], remove_stop_words: bool = True, stemming: bool = True) -> List[str]:
    # same pipeline for documents and queries, so both end up in the same term space
    tokens = tokenize(normalize_text(text))
    if remove_stop_words:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    if stemming:
        # stemmed terms are what is stored in the index
        tokens = [STEMMER.stem(t) for t in tokens]
    return tokens


def details_to_text(product_details: Optional[Dict[str, str]]) -> str:
    if not product_details:
        return ""
    # only the values are kept 
    values = [str(product_details[k]) for k in DETAIL_KEYS if product_details.get(k)]
    # for Yes/No attributes the value carries no information
    values += [k for k in BOOLEAN_DETAIL_KEYS if str(product_details.get(k, "")).lower() == "yes"]
    return " ".join(values)


def clean_seller(seller: Optional[str]) -> Optional[str]:
    if not seller:
        return None
    return SELLER_ARTIFACT.sub("", seller).strip() or None


def compute_discount(doc: Document) -> Optional[float]:
    if doc.discount is not None:
        return doc.discount
    # the guards avoid dividing by zero and handle missing prices 
    if doc.actual_price and doc.selling_price is not None and doc.actual_price > 0:
        return round(100 * (1 - doc.selling_price / doc.actual_price), 1)
    return None


def preprocess_document(doc: Document) -> dict:
    record = {field: getattr(doc, field) for field in OUTPUT_FIELDS}
    record["seller"] = clean_seller(doc.seller)
    record["discount"] = compute_discount(doc)
    # no rating is not a bad rating
    record["has_rating"] = doc.average_rating is not None

    record["title_terms"] = build_terms(doc.title)
    record["description_terms"] = build_terms(doc.description)
    record["details_terms"] = build_terms(details_to_text(doc.product_details))
    record["brand_terms"] = build_terms(doc.brand, stemming=False)  # brand names are proper nouns, do not stem them
    record["sub_category_terms"] = build_terms(doc.sub_category)
    # category is not indexed 
    # and neither is seller (users do not search by it); both stay available as plain attributes
    record["terms"] = (record["title_terms"] + record["description_terms"] + record["details_terms"]
                       + record["brand_terms"] + record["sub_category_terms"])
    return record


def preprocess_corpus(corpus: Dict[str, Document]) -> pd.DataFrame:
    records = [preprocess_document(doc) for doc in corpus.values()]
    # pid is both the index (fast lookup by pid) and a column (kept for output)
    return pd.DataFrame(records).set_index("pid", drop=False).rename_axis(None)
