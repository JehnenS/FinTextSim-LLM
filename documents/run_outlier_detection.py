import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import os

os.chdir("/mnt/sdb1/home/simonj") #set working directory

#------------load data
file_loc = "paper2/Data/Text/10-K/item7/item7_text.pkl"

with open(file_loc, "rb") as file:
    result_dict = pickle.load(file)

item7_texts = result_dict["item7_texts"]
item7_metadata = result_dict["item7_metadata"]

print(f"Number of documents: {len(item7_texts)}")
print(f"Number of metadata: {len(item7_metadata)}")

from documents.OutlierDetector import OutlierDetector

detector = OutlierDetector(
    text_list = item7_texts,
    metadata_list = item7_metadata
)

detector.run_quantile(
    min_words_threshold = 250,
    quantile_threshold = 0.99,
    output_dir = "paper2/Data/Text/10-K/item7/item7_text_outlier.pkl"
)