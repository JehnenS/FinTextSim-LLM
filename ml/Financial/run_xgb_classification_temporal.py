import pandas as pd
import numpy as np
import cupy as cp
from tqdm import tqdm
import os
import pickle


os.chdir("/mnt/sdb1/home/simonj") #set working directory

import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to config JSON file")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = json.load(f)

# Now you can access config["target_variable"], etc.
print(f"Running {__name__} with config:", config)



results_dict = {}

results_dict["config"] = config

print(f"\nSample: {results_dict["config"].get("sample")}")
print(f"Outlier detection method: {results_dict["config"].get("outlier_detection")}")
print(f"Oversampling: {results_dict["config"].get("oversample_method")}")
print(f"Start of test-period: {results_dict["config"].get("test_year_start")}")
print(f"Feature Set: {results_dict["config"].get("feature_set")}")
print(f"Exclude quarter features: {results_dict["config"].get("exclude_quarter_features")}")
print(f"Target variable: {results_dict["config"].get("target_variable")} - {results_dict["config"].get("target_period")}")
print(f"Adjustment of target variable: {results_dict["config"].get("adjust_target_variable")}\n")

from ml.utils_ml import load_result_loc, load_data_loc, load_sentiment_loc

result_loc = load_result_loc(results_dict, model_name = "xgb_temporal")

#-----------Load data
data_loc = load_data_loc(results_dict)

with open(data_loc, "rb") as file:
    data = pickle.load(file)

data.keys()

results = data["results"]
results_economic = results["economic"] #extract economic data to make it cleanly ticker-based
results.pop("economic", None) #remove economic from results to make it cleanly ticker-based

#----------get target
from feature_creation.FinTargetExtractor import FinTargetExtractor


ft_extractor = FinTargetExtractor(
    result_dict = results,
    target_table_name = "income_growth", #table in which the target variable can be found
    target_variable_name = results_dict["config"].get("target_variable"), #name of the target variable
    target_period = results_dict["config"].get("target_period"), #can be either FY, Q1, Q2, Q3, Q4
    min_year = results_dict["config"].get("min_year_target"), #year of reported values --> in df: min_year -1 as we need the 2011 predictors to predict 2012 results --> shift by -1 so that 2012 results align with 2011 predictors for merging
    max_year = results_dict["config"].get("max_year_target"),
    kpi_min_abs_value = results_dict["config"].get("kpi_min_abs_value"), 
    kpi_max_abs_value = results_dict["config"].get("kpi_max_abs_value"),
    binary_label = results_dict["config"].get("binary_label"), #boolean for transformation of label into binary classes
    adjust_variable = results_dict["config"].get("adjust_target_variable")
)

target_df = ft_extractor.get_target_df()



#---------Load features
#define the paths
swade_path = "paper2/Data/Features/swade_features_fd_sp500.csv"
fin_path = "paper2/Data/Features/fin_features_fd_sp500.csv"

#text_llm_stock_path = "paper3/Features/sentiment_llm_stock.csv"
#text_llm_fin_path = "paper3/Features/sentiment_llm_fin.csv"

swade_features = pd.read_csv(swade_path)
fin_features = pd.read_csv(fin_path)

#decide if we want to include quarter-based features
if results_dict["config"].get("exclude_quarter_features"):
    quarter_suffixes = ("_Q1", "_Q2", "_Q3", "_Q4")
    
    quarter_based_features = [
        col for col in fin_features.columns
        if col.endswith(quarter_suffixes)
    ]
    print(f"Number of quarter-based features: {len(quarter_based_features)}")
    fin_features = fin_features[[col for col in fin_features.columns if col not in quarter_based_features]]
    print(f"Shape of df without quarter-based features: {fin_features.shape}")


#assign financiancials based on feature set
if results_dict["config"].get("feature_set") == "swade":
    financials = swade_features
else:
    financials = fin_features


text_transformer_path = "paper2/Data/Features/text_features/sentiment_transformers_fd_sp500_quantile_temporal.pkl" 
#get the text features for the relevant text features
with open(text_transformer_path, "rb") as file:
    textual_data = pickle.load(file)

text_features_transformer_fin = textual_data[results_dict["config"].get("fintextsim")]["text_features_fin"]
text_features_transformer_stock = textual_data[results_dict["config"].get("fintextsim")]["text_features_stock"]

text_rag_path = "paper2/Data/Features/text_features/sentiment_rag_fd_sp500_quantile_temporal.pkl" 
#get the text features for the relevant text features
with open(text_rag_path, "rb") as file:
    textual_data_rag = pickle.load(file)

text_features_rag_fin = textual_data_rag[results_dict["config"].get("fintextsim")]["text_features_fin"]
text_features_rag_stock = textual_data_rag[results_dict["config"].get("fintextsim")]["text_features_stock"]

#--------------------LLM text features
from ml.utils_ml import models


#----create stock features
from ml.MLDatasetBuilderCSV import MLDatasetBuilder
from ml.utils_ml import categorical_cols, cols_to_exclude, text_only_cols
from preprocess.PrepML import PrepML


#remove all other text features to iterate only over basic features
for key in ['azimi', 'huang', 'zhang', 'gupta_a', 'gupta_b', 'gupta_c', 'renault', 'vamossy']:
    text_features_transformer_fin.pop(key, None)


for feature_set in text_features_transformer_fin.keys():
    print(f"\nFeature set: {feature_set}\n")
    results_dict.setdefault("results", {})              # Ensure 'results' exists
    results_dict["results"].setdefault(feature_set, {}) # Ensure the feature_set dict exists

    #generate ML datasets for different textual feature sets
    builder = MLDatasetBuilder(
        target_df = target_df,
        financials = financials,
        text_features_transformer = text_features_transformer_fin[feature_set],
        text_features_llm = text_features_rag_fin[feature_set]
    )
    
    df_fin, df_fin_mean_sentiment, df_fin_text, df_fin_llm = builder.build_all() #df_text_only
    df_text_only = df_fin_text[text_only_cols] #create text only df
    print(f"df_text_only: {df_text_only.shape}")

    dfs = [df_fin_text, df_fin, df_fin_mean_sentiment, df_fin_text, df_fin_llm] #set df_fin_text as base for text only df --> drop same rows based on financial features; then filter columns in X_train and X_test for sentiment features
    df_names = ["text_only", "fin", "fin_mean_sentiment", "fin_text_transformer", "fin_text_llm"] #"text_only",
        
    #clean the dfs if a df is None
    dfs = [df for df in dfs if df is not None]
    df_names = [name for df, name in zip(dfs, df_names) if df is not None]

    for i, df in enumerate(dfs):
        print(f"\n-------------DF: {df_names[i]}---------------------")
        mlprep = PrepML(
            df = df,
            target_name = "target"
        )
    
        X_train, X_test, y_train, y_test, neg_instances, pos_instances, X, y = mlprep.run_tree_preprocessing(
            categorical_cols = categorical_cols,
            cols_to_exclude = cols_to_exclude,
            threshold_columns = 0.5,
            threshold_rows = 0.4,
            test_start_year = results_dict["config"].get("test_year_start"),
            oversample_method = results_dict["config"].get("oversample_method")
        )

        if df_names[i] == "text_only":
            X_train = X_train[text_only_cols]
            X_test = X_test[text_only_cols]
            print(f"Shape of X_train text only: {X_train.shape}")
            print(f"Shape of X_test text only: {X_test.shape}")
        
        #--------train and evaluate
        from ml.Classifier import Classifier
        
        classifier = Classifier(
            X_train, X_test, y_train, y_test, neg_instances, pos_instances, X, y
        )
        
        y_train_pred, y_test_pred, y_train_prob, y_test_prob = classifier.xgb_classification(
            max_depth = 2, #1-4 in Chen et al. 2022
            min_child_weight = 10, #10 in Chen et al. 2022
            learning_rate = 0.01, #learning rates of 0.005, 0.01 and 0.05 in Chen et al. 2022
            subsample = 0.8, #0.5 in Chen et al. 2022
            colsample_bytree = 1.0, 
            num_parallel_tree = 1, #num_parallel_tree to 1 to avoid triggering RF behavior
            num_boost_round = 1000, #500-2000 in Chen et al. 2022
            reg_lambda = 1.5,
            reg_alpha = 0.0,
            perform_cv = False,
            scale_pos_weight_factor = 1.0
        )
        
        #evaluate
        #evaluate
        print("\nTrain Set:")
        train_results, train_results_filtered = classifier.evaluate_model(
            y_true = y_train,
            y_pred = y_train_pred,
            y_prob = y_train_prob,
            n_bootstrap_auc = 0
        )
        
        print("\nTest Set:")
        test_results, test_results_filtered = classifier.evaluate_model(
            y_true = y_test,
            y_pred = y_test_pred,
            y_prob = y_test_prob,
            n_bootstrap_auc = results_dict["config"].get("n_bootstrap_auc")
        )
        
        
        results_dict["results"][feature_set][df_names[i]] = {
            'train_results': train_results,
            'train_results_filtered': train_results_filtered,
            "test_results": test_results, 
            "test_results_filtered": test_results_filtered,
            #"model": model,
            #"shap_values": shap_values,
            #"explainer": explainer,
            #"shap_interaction_values": shap_interaction_values,
            "X_train": X_train,
            "X_test": X_test,
            "feature_names": list(X.columns)
        }



with open(result_loc, "wb") as file:
    pickle.dump(results_dict, file)
print(f"Data saved to {result_loc}")