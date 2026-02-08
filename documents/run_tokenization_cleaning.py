import pickle
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

"""
Transform the outlier-cleaned documents into sentence format
Transform the metadata correspondingly
clean the documents and sentences
"""

os.chdir("/mnt/sdb1/home/simonj") #set working directory

result_file_loc = "paper2/Data/Text/10-K/item7/item7_text_outlier_sentences_clean.pkl"

#------------load data
file_loc = "paper2/Data/Text/10-K/item7/item7_text_outlier.pkl"

with open(file_loc, "rb") as file:
    result_dict = pickle.load(file)

item7_texts = result_dict["item7_texts"]
item7_metadata = result_dict["item7_metadata"]

print(f"Number of documents: {len(item7_texts)}")
print(f"Number of metadata: {len(item7_metadata)}")

#---------------------
from labeled_dataset.utils_labeled_dataset import substitutions, replace_subs, clean_sentence

#replace substitutions in whole documents
item7_texts_clean = [replace_subs(doc, substitutions) for doc in tqdm(item7_texts, desc = "Substitution replacements")]

#tokenize text into sentences
from nltk import sent_tokenize

item7_texts_sentences = [(idx, sent) for idx, doc in enumerate(item7_texts_clean) for sent in sent_tokenize(doc)]  #list of doc index and text per sentence

#clean sentences - trailing whitespaces, punctuation, etc.
item7_texts_sentences = [(idx, clean_sentence(sent)) for (idx, sent) in tqdm(item7_texts_sentences, desc = "Clean sentences (trailing whitespaces, punctuation)")]

# Filter sentences based on length (number of words) using split
item7_texts_sentences_id = [(idx, sent) for idx, sent in tqdm(item7_texts_sentences, desc = "Filtering by sentence length") if 5 < len(sent.split()) < 50] #filter list of tuples
rel_docs_after_filter, item7_texts_clean_sentences = zip(*item7_texts_sentences_id) #extract id and text separately

print(f"number of sentences before removing based on word number: {len(item7_texts_sentences)}")
print(f"number of sentences after removing based on word number: {len(item7_texts_clean_sentences)}")


#---------------handle metadata
rel_docs_after_filter_unique = list(set(rel_docs_after_filter))

item7_metadata_rel = [meta for idx, meta in tqdm(enumerate(item7_metadata), desc = "Filter doc metadata") if idx in rel_docs_after_filter_unique]

from collections import Counter
index_counts = Counter(rel_docs_after_filter) #number of sentences per document

# Expand metadata based on rel docs --> assign metadata to each sentence
item7_metadata_expanded = [
    {
        **item,                  # unpack original metadata dict
        'doc_id': idx,          # add document ID
        'sentence_id': sent_id  # add sentence ID within document
    }
    for idx, item in zip(rel_docs_after_filter_unique, item7_metadata_rel)
    for sent_id in range(index_counts[idx])
]

#expanded metadata contains:[(doc_id, yearmonthday, year, month, cik, company, sector, query, sentence_id (per doc_id)]
print(f"Number of metadata entries: {len(item7_metadata_expanded)}")
print(f"Number of sentences: {len(item7_texts_clean_sentences)}")



result_dict = {
    "item7_texts": item7_texts_clean_sentences,
    "item7_metadata": item7_metadata_expanded
}


with open(result_file_loc, "wb") as file:
    pickle.dump(result_dict, file)

print(f"Results saved to {result_file_loc}")