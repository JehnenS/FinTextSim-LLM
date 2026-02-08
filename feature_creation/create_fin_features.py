import pandas as pd
import numpy as np
import pickle
import os

os.chdir("/mnt/sdb1/home/simonj")

### Define save loc
save_loc = "paper2/Data/Features/fin_features_fd_sp500.csv"
save_loc_swade = "paper2/Data/Features/swade_features_fd_sp500.csv"

#---------Load data
data_loc = "paper2/Data/FMP/fmp_data.pkl"

with open(data_loc, "rb") as file:
    data = pickle.load(file)

results = data["results"]
results_economic = results["economic"] #extract economic data to make it cleanly ticker-based
results.pop("economic", None) #remove economic from results to make it cleanly ticker-based

#--------create features for growth based input - quarter + FY
from feature_creation.FeatureCreator import FeatureCreator
fc = FeatureCreator(results)
fy_features, quarter_features = fc.run()

#-------create features for absolute based input
fc = FeatureCreator(results, input_table_names = ["income_statement", "balance_sheet", "cash_flow"], prefixes=["is_abs_", "bs_abs_", "cf_abs_"])
fy_features_abs = fc.run_fy()

#merge growth, absolute features and quarter features
full_features = fy_features.merge(fy_features_abs, on = ["ticker", "year"], how = "outer", suffixes = ("_rel", "_abs"))
full_features = full_features.merge(quarter_features, on = ["ticker", "year"], how = "outer")


#--------generate stock-based features based on Swade et al.
from feature_creation.SwadeFinFeatureCreator import SwadeFinFeatureCreator
from feature_creation.FilingDateExtractor import FilingDateExtractor

extractor = FilingDateExtractor(
    result_dict = results
)
#extract filing dates + sanity check-filtering
filing_dates = extractor.run(remove_duplicates = True, max_delay_dates = 180)


#generate stock features from the raw stock data
creator = SwadeFinFeatureCreator()
stock_df = creator.extract_rows(results)

stock_features = creator.run_stock_features(stock_df, filing_dates)

#-------------generate economic-based features
from feature_creation.EconomicFeatureCreator import EconomicFeatureCreator

creator = EconomicFeatureCreator(results_economic)
economic_df = creator.run(filing_dates, date_name = "FY")

#-------------------Merging and cleaning
#merge with other features
full_features_incl_stock = full_features.merge(stock_features, on = ["ticker", "year"], how = "outer")
full_features_incl_stock = full_features_incl_stock.merge(economic_df, on = ["ticker", "year"], how = "outer")


#create additional Swade et al. features
full_features_incl_stock["market_leverage"] = full_features_incl_stock["bs_abs_totalDebt"] / (full_features_incl_stock["bs_abs_totalDebt"] + full_features_incl_stock["km_marketCap"])
full_features_incl_stock["gp_to_assets"] = full_features_incl_stock["is_abs_grossProfit"] / full_features_incl_stock["bs_abs_totalEquity"]
full_features_incl_stock["net_income_to_equity"] = full_features_incl_stock["is_abs_netIncome"] / full_features_incl_stock["bs_abs_totalEquity"]
full_features_incl_stock["net_income_to_assets"] = full_features_incl_stock["is_abs_netIncome"] / full_features_incl_stock["bs_abs_totalAssets"]
full_features_incl_stock["share_turnover"] = full_features_incl_stock["mean_volume"] / full_features_incl_stock["is_abs_weightedAverageShsOut"]

#---------Create Swade features
from feature_creation.utils_feature_creation import swade_features, dups
rel_cols = ["ticker", "year"] + swade_features

swade_features = full_features_incl_stock[rel_cols]

#------Remove absolute features and duplicates from other feature set
dups_full = [
    col
    for base in dups
    for col in [base, f"{base}_Q1", f"{base}_Q2", f"{base}_Q3", f"{base}_Q4"]
    if col in full_features_incl_stock.columns
]

print(f"Number of duplicate features incl. quarters: {len(dups_full)}") #--> more for quarters --> Q1, Q2, Q3, Q4 metrics

abs_features = [col for col in fy_features_abs if col not in ["ticker", "year"]]
print(f"Number of absolute features: {len(abs_features)}")

irrel_features = abs_features + dups_full

full_features_incl_stock_clean = full_features_incl_stock[[col for col in full_features_incl_stock if col not in irrel_features]]


#--------Save results
full_features_incl_stock_clean.to_csv(save_loc, index = False)
swade_features.to_csv(save_loc_swade, index = False)
print(f"Financial features saved to {save_loc}")
print(f"Shape of financial feature df: {full_features_incl_stock_clean.shape}")
print(f"Swade features saved to {save_loc_swade}")
print(f"Shape of Swade feature df: {swade_features.shape}")