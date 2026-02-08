import pandas as pd
import numpy as np
import pickle
import os

os.chdir('/mnt/sdb1/home/simonj')

result_file = "paper2/Data/Features/text_features/sentiment_transformers_fd_sp500_quantile.pkl" 


#------load data
loc = "paper2/Data/FMP/cik_ticker_info.pkl"

with open(loc, "rb") as file:
    data = pickle.load(file)
data.keys()

historical_constituents_sp500 =  data["historical_constituents_sp500"]
cik_ticker_info = data["cik_ticker_info"]

print(f"Number of found cik-ticker combinations: {len(cik_ticker_info)}")


result_loc = "paper2/Data/FMP/sp500_ticker_list.pkl"

with open(result_loc, "rb") as file:
    data = pickle.load(file)

data.keys()
ticker_list = data["ticker_list"]

cik_ticker_info_sp500 = [entry for entry in cik_ticker_info if entry.get("symbol") in ticker_list]
print(f"Number of entries: {len(cik_ticker_info_sp500)}")

# CIK → ticker mapping
cik_to_ticker = {entry["cik"]: entry["symbol"] for entry in cik_ticker_info_sp500}


#-------------Load knn data to get the model names to iterate over
knn_loc = "paper2/Results/knn/knn_centroid_fd_sp500_quantile.pkl"

with open(knn_loc, "rb") as file:
    knn_data = pickle.load(file)



#-----------------Prep the data which remains the same across all models
from preprocess.PrepBasics import DataPreprocessor

prep = DataPreprocessor()

texts, meta = prep.load_metadata("paper2/Data/Text/item7_text_rel_tickers_quantile.pkl", text_name = "item7_texts", metadata_name = "item7_metadata")
sentiment_assignments, fls_results = prep.load_modalities("paper2/Data/Modalities/modalities_fd_sp500_quantile.pkl")

results = {}
from feature_creation.TextFeatureCreatorTransformer import TextFeatureCreatorTransformer

#iterate over all model assignments
for model_name in knn_data.keys():
    print(f"\n----------Model name: {model_name}----------------")
    indices = prep.load_knn_indices(knn_loc, model_name = model_name)
    #indices_to_check, answers = prep.load_rag_results()
    
    #print(f"Number of answers: {len(answers)}")
    #indices, noise = transform_rag_output(indices, indices_to_check, answers, topics)
    
    #df_grouped = prep.build_grouped_df(meta_merged, texts, indices, topics)
    
    
    
    creator = TextFeatureCreatorTransformer(sentiment_assignments, indices[:, 0], meta, cik_to_ticker)
    text_features_stock = creator.run_all_stock(filter_neutral_scores = False)
    text_features_fin = creator.run_all_fin(filter_neutral_scores = False)
    #print(f"Shape of df stock: {text_features_stock.shape}")
    #print(f"Shape of df fin: {text_features_fin.shape}")

    
    #from feature_creation.utils_feature_creation import extract_topic_sentiments
    
    #sentiment_dict = extract_topic_sentiments(text_features_stock)
    
    results[model_name] = {
        "text_features_stock": text_features_stock,
        "text_features_fin": text_features_fin,
     #   "sentiment_dict": sentiment_dict
    }

with open(result_file, "wb") as file:
    pickle.dump(results, file)
    
print(f"Results saved to {result_file}")