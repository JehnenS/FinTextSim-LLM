import pandas as pd
import numpy
import pickle
import os

os.chdir("/mnt/sdb1/home/simonj") #set working directory


#--------------------------------Load documents

from paper2.documents.DocLoader import DocLoader

loader = DocLoader(
    rel_cik_list = None, #run extraction for ALL publicly traded companies
    zip_path = "10-K",
    keyword = "_10-K_",
    output_dir = "paper2/Data/Text/10-K/"
)

loader.run_extraction()