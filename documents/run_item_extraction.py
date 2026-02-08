import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle
import os

os.chdir("/mnt/sdb1/home/simonj") #set working directory

matches_pattern = r'(part\s+ii\b|part\s+iii\b|items?\s(7[\.\:\s])|item\s(8[\.\:\s])|discussion\sand\sanalysis\sof\s(consolidated\sfinancial|financial)\scondition|(consolidated\sfinancial|financial)\sstatement(s\sand\ssupplementary\sdata))'

from documents.Item7Extractor import ItemExtractor

extractor = ItemExtractor(
    matches_pattern = matches_pattern,
    file_loc= "paper2/Data/Text/10-K",
    output_dir = "paper2/Data/Text/10-K/item7/item7_text.pkl"
)

extractor.run()