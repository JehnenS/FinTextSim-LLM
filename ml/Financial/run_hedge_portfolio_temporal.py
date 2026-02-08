import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

from ml.utils_ml import load_result_loc, load_data_loc, load_sentiment_loc

os.chdir("/mnt/sdb1/home/simonj")
result_loc = "paper2/Results/ML/Financials/hedge_portfolio/hedge_portfolio_temporal.pkl"

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
#results_dict["config"] = {
 # "target_period": "FY",
  #"kpi_min_abs_value": 0.01,
  #"kpi_max_abs_value": 200.0,
  #"binary_label": True,
  #"target_variable": "growthEPSDiluted",
  #"exclude_quarter_features": True,
  #"fintextsim": "acl_modern_bert",
  #"adjust_target_variable": True,
  #"sample": "fd_sp500",
  #"feature_set": None,
  #"outlier_detection": "quantile",
  #"oversample_method": None,
  #"test_year_start": 2020,
  #"min_trading_days": 210,
  #"value_weighting": False,
  #"min_year_target": 1990,
  #"max_year_target": 2030,
  #  "long_threshold": 0.6,
   # "short_threshold": 0.4,
#}






print(f"\nLong threshold: {results_dict["config"].get("long_threshold")}")
print(f"Short threshold: {results_dict["config"].get("short_threshold")}\n")

#------------Load data
data_loc = "paper2/Data/FMP/fmp_data.pkl"

with open(data_loc, "rb") as file:
    data = pickle.load(file)

data.keys()

results = data["results"]
results_economic = results["economic"] #extract economic data to make it cleanly ticker-based
results.pop("economic", None) #remove economic from results to make it cleanly ticker-based
print(f"Keys from ticker-based results: {results.keys()}")
#print(f"Keys from economic results: {results_economic.keys()}")

#------------create targets
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
target_df

#-----------create filing df
from feature_creation.FilingDateExtractor import FilingDateExtractor

extractor = FilingDateExtractor(
    result_dict = results
)

filing_df = extractor.extract_data(remove_duplicates = True)

#---------Load and prepare SP500 data
loc = "paper2/Data/FMP/fmp_market_data.pkl"

with open(loc, "rb") as file:
    data = pickle.load(file)

sp500_daily = data["sp500_daily"]["historical"]
sp500_daily_df = pd.DataFrame(sp500_daily).sort_values("date").reset_index(drop = True)
sp500_daily_df["return"] = sp500_daily_df["adjClose"].pct_change()
sp500_daily_df["date"] = pd.to_datetime(sp500_daily_df["date"])

#-----------
fintextsim_version = "acl_modern_bert"
exclude_quarters = True
text_features = "basic"

#---------Load features and prepare ML datasets
#define the paths
swade_path = "paper2/Data/Features/swade_features_fd_sp500.csv"
fin_path = "paper2/Data/Features/fin_features_fd_sp500.csv"


text_transformer_path = "paper2/Data/Features/text_features/sentiment_transformers_fd_sp500_quantile_temporal.pkl" 

swade_features = pd.read_csv(swade_path)
fin_features = pd.read_csv(fin_path)

#get the text features for the relevant text features
with open(text_transformer_path, "rb") as file:
    textual_data = pickle.load(file)

text_features_transformer_fin = textual_data[fintextsim_version]["text_features_fin"][text_features]
text_features_transformer_stock = textual_data[fintextsim_version]["text_features_stock"][text_features]

text_rag_path = "paper2/Data/Features/text_features/sentiment_rag_fd_sp500_quantile_temporal.pkl" 
#get the text features for the relevant text features
with open(text_rag_path, "rb") as file:
    textual_data_rag = pickle.load(file)

text_features_rag_fin = textual_data_rag[results_dict["config"].get("fintextsim")]["text_features_fin"][text_features]
text_features_rag_stock = textual_data_rag[results_dict["config"].get("fintextsim")]["text_features_stock"][text_features]


#decide if we want to include quarter-based features
exclude_quarters = True
if exclude_quarters:
    quarter_suffixes = ("_Q1", "_Q2", "_Q3", "_Q4")
    
    quarter_based_features = [
        col for col in fin_features.columns
        if col.endswith(quarter_suffixes)
    ]
    print(f"Number of quarter-based features: {len(quarter_based_features)}")
    fin_features = fin_features[[col for col in fin_features.columns if col not in quarter_based_features]]
    print(f"Shape of df without quarter-based features: {fin_features.shape}")


from ml.MLDatasetBuilderCSV import MLDatasetBuilder
from ml.utils_ml import text_only_cols

builder = MLDatasetBuilder(
    target_df = target_df,
    financials = fin_features,
    text_features_transformer = text_features_transformer_fin,
    text_features_llm = text_features_rag_fin
)

df_fin, df_fin_mean_sentiment, df_fin_text, df_fin_llm = builder.build_all()
df_text_only = df_fin_text[text_only_cols] #create text only df
print(f"df_text_only: {df_text_only.shape}")


#-----------------

from ml.HedgePortfolio import HedgePortfolio


hp = HedgePortfolio(
    ml_dataset = df_fin_text,
    feature_set = "basic"
)

model_names = ["lr_temporal", "rf_temporal", "xgb_temporal"]
df_names = ["text_only", "fin", "fin_mean_sentiment", "fin_text_transformer", "fin_text_llm"]


#for df_name in tqdm(df_names, desc = "DF progress"):
 #   for model in tqdm(model_names, desc = "Model Progress"):
  #      print(f"\n\n----------------df: {df_name} - ML Model: {model}")

   #     results_hedge, results_hedge_pred, results_pf = hp.run(
    #        ml_config = results_dict["config"],
     #       model_name = model,
      #      result_dict = results,
       #     market_df_daily = sp500_daily_df,
        #    short_threshold = results_dict["config"].get("short_threshold"),
         #   long_threshold = results_dict["config"].get("long_threshold"),
          #  df_name = df_name,
           # filing_df = filing_df,
            #min_trading_days = min_trading_days,
            #value_weighted = results_dict["config"].get("value_weighting")
        #)



shift_years = [0, 1, 2, 3]
if "results" not in results_dict:
    results_dict["results"] = {}

#iterate over all shift years
for shift_year in shift_years:
    #construct return df
    return_base = hp.run_cumulative_return_base(
        ml_config = results_dict["config"], #dictionary of ML configuration --> used to construct the path to the results
        model_name = "lr_temporal", #name of the ML model for which we want to load the results --> lr, rf or xgb (potentially with _detrend) --> not relevant in this case as we only create the base and update it in the iteration over models
        result_dict = results, #dictionary with results stored by ticker
        market_df_daily = sp500_daily_df, #df with daily market information 
        filing_df = filing_df, #df with filing dates/end of period of reports and fiscal year per ticker, year combination
        df_name = df_names[-2],
        min_trading_days = results_dict["config"].get("min_trading_days"),
        shift_year = shift_year #assign shft year
    )
    if shift_year not in results_dict["results"]:
        results_dict["results"][shift_year] = {}
    
    results_dict["results"][shift_year] = {
        "return_df": return_base,
    }
    
    #iterate over models and dataframes
    for model in model_names:
        print(f"\n\n\n##################Model: {model}\n")
        for df in df_names:
            print(f"\n------------Feature set: {df}\n")
            results_hedge, result_hedge_pred, results_hedge_fp = hp.load_model_probs_and_run_eval(
                return_df = return_base, # df created by run_cumulative_return_base
                ml_config = results_dict["config"], #dictionary of ML configuration --> used to construct the path to the results 
                model_name = model, #name of the ML model for which we want to load the results --> lr, rf or xgb (potentially with _detrend)
                long_threshold = results_dict["config"].get("long_threshold"), #threshold to assign long position in portfolio --> if x > threshold: long  
                short_threshold = results_dict["config"].get("short_threshold"), #threshold to assign short position in portfolio --> if x < threshold: short)
                df_name = df,
                value_weighted = results_dict["config"].get("value_weighting"),
            )
            if model not in results_dict["results"][shift_year]:
                results_dict["results"][shift_year][model] = {}
            results_dict["results"][shift_year][model][df] = {
                "hedge_threshold": results_hedge,
                "hedge_pred": result_hedge_pred,
                "hedge_perfect_foresight": results_hedge_fp
            }


with open(result_loc, "wb") as file:
    pickle.dump(results_dict, file)
print(f"Results saved to {result_loc}")