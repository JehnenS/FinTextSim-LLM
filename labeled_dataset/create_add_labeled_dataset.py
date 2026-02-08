import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

os.chdir("/mnt/sdb1/home/simonj")

from labeled_dataset.utils_labeled_dataset import topic_names, keywords, exclusion_dict, keyword_blacklist_substring

result_file_loc = "paper2/Data/add_labeled_dataset/add_labeled_dataset.pkl"
print(f"Result loc: {result_file_loc}")


#----------Load data
loc = "paper2/Data/add_labeled_dataset/full_texts_add_data.pkl"

with open(loc, "rb") as file:
    results = pickle.load(file)

texts = results["texts"]
print(f"Number of texts: {len(texts)}")


#----------------------------
from labeled_dataset.LabeledDatasetCreator import LabeledDatasetCreator

creator = LabeledDatasetCreator(
    sentences = texts,
    topic_names = topic_names,
    keyword_list = keywords,
    keyword_blacklist = keyword_blacklist_substring,
    exclusion_dict = exclusion_dict
)

ld_results = creator.run()


#--------save results
with open (result_file_loc, "wb") as file:
    pickle.dump(ld_results, file)

print(f"Additional labeled dataset saved to: {result_file_loc}")