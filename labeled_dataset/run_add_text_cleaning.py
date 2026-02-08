import pandas as pd
import pickle
import os

os.chdir("/mnt/sdb1/home/simonj")

result_loc = "paper2/Data/add_labeled_dataset/full_texts_add_data.pkl"
#----------Load dataset and prepare it
from datasets import load_dataset

dataset_file = "lemousehunter/SnP500-annual-and-sustainability-reports"
add_dataset = load_dataset(dataset_file)
add_dataset = add_dataset["train"]

add_dataset_rel = add_dataset \
    .map(lambda row: {"year": int(row["year"])}) \
    .filter(lambda row: row["year"] >= 2000)  # Filter rows from 2000 onward

add_texts = add_dataset_rel["chunk_list"]
print(f"Number of add texts (not in sentences yet): {len(add_texts)}")

# Flatten the list of lists into a single list of strings
flat_add_texts = [text for sublist in add_texts for text in sublist]

#-----------clean the sentence input
from labeled_dataset.TextCleaner import TextCleaner
from labeled_dataset.utils_labeled_dataset import substitutions

cleaner = TextCleaner(flat_add_texts)

clean_texts = cleaner.run(substitutions = substitutions)

results = {
    "texts": clean_texts
}

with open(result_loc, "wb") as file:
    pickle.dump(results, file)

print(f"Results saved to {result_loc}")