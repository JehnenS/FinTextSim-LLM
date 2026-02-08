import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

class FeatureCreator:
    def __init__(
        self, 
        result_dict,
        yearly_cat="financials",
        quarterly_cat="quarterly_financials",
        input_table_names=["key_metrics", "ratios", "cashflow_growth", "income_growth", "balance_sheet_growth", "financial_growth"],
        prefixes=["km_", "ratios_", "cf_", "is_", "bs_", "fg_"]
    ):
        self.result_dict = result_dict
        self.yearly_cat = yearly_cat
        self.quarterly_cat = quarterly_cat
        self.input_table_names = input_table_names
        self.features = {}  # store dfs by table name
        self.prefixes = prefixes
        self.features_quarterly = {}  # store quarterly dfs
        self.full_df = None
        self.full_df_quarterly = None

    def extract_features_fy(self, 
                            input_table_name, 
                            prefix,
                            min_year=1990, 
                            max_year=2025):
        """
        Extracts all numeric features from a financial table for each ticker and year --> e.g. balance_sheet_growth.
        """
        #create list to store results per row (equals ticker-year combo)
        rows = []

        #iterate over each ticker
        for ticker, data in tqdm(self.result_dict.items(), desc=f"Extracting from {input_table_name}"):
            table = data.get(self.yearly_cat).get(input_table_name, [])  # extract the relevant table

            #iterate over entries in the relevant table
            for entry in table:
                year = entry.get("calendarYear") #--> corresponds to the fiscal year: e.g. AAPL: 'date': '2024-12-28','symbol': 'AAPL', calendarYear': '2025',  'period': 'Q1', --> Q1 of fiscal year 2025 even though quarter end date lies in 2024
                if year is None:
                    continue
                try:
                    year = int(year)
                except ValueError:
                    continue
                period = entry.get("period")

                #ensure relevant timeframe and focus only on full periods instead of quarters
                if not year or not (min_year <= year <= max_year) or period != "FY":
                    continue

                #append ticker and year to the row
                row = {"ticker": ticker, "year": year}

                #iterate over all entries in the table --> key-value pairs
                for key, value in entry.items():
                    #skip metadata --> do not treat them as numerical features
                    if key in {"date", "symbol", "calendarYear", "period"}:
                        continue
                    #only keep numeric (integer or float) values and append them to row
                    if isinstance(value, (int, float)):
                        col_name = f"{prefix or ''}{key}"
                        row[col_name] = value

                rows.append(row)

        df = pd.DataFrame(rows).sort_values(["ticker", "year"]).reset_index(drop=True)

        # remove duplicates
        df.drop_duplicates(subset=["ticker", "year"], inplace=True)

        return df

    def extract_features_quarter(self, input_table_name, prefix, min_year=1990, max_year=2025):
        """
        Extract quarterly numeric features per ticker-year-quarter.
        """
        rows = []
        for ticker, data in tqdm(self.result_dict.items(), desc=f"Extracting Quarterly from {input_table_name}"):
            table = data.get(self.quarterly_cat, {}).get(input_table_name, [])
            for entry in table:
                year = entry.get("calendarYear")
                if year is None:
                    continue
                try:
                    year = int(year)
                except ValueError:
                    continue
                period = entry.get("period")
                if not (min_year <= year <= max_year) or not period.startswith("Q"):
                    continue

                #append ticker and year to row
                row = {"ticker": ticker, "year": year, "period": period}

                #iterate over all key, entry pairs in the relevant table
                for key, value in entry.items():
                    #skip metadata
                    if key in {"date", "symbol", "calendarYear", "period"}:
                        continue
                    #append only numerical values (float, integer) to row, i.e. df
                    if isinstance(value, (int, float)):
                        row[f"{prefix}{key}"] = value
                rows.append(row)

        df = pd.DataFrame(rows).sort_values(["ticker", "year", "period"]).reset_index(drop=True)
        df.drop_duplicates(subset=["ticker", "year", "period"], inplace=True)

        return df

    def run_fy(self, min_year=1990, max_year=2025):
        """
        Run extraction for yearly (FY) data.
        """
        merged_df = None
        for i, table in tqdm(enumerate(self.input_table_names), desc="FY Table progress", total=len(self.input_table_names)):
            df = self.extract_features_fy(table, self.prefixes[i], min_year, max_year)
            self.features[table] = df

            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on=["ticker", "year"], how="outer")

        self.full_df = merged_df.sort_values(["ticker", "year"]).reset_index(drop=True)
        return self.full_df


    def pivot_quarterly_features(self, df, chunk_size=50):
        """
        Pivot the quarterly features into wide format --> transform e.g. revenueGrowth Q1-Q4 
        from 4 rows 1 column to 1 row 4 columns --> revenueGrowth_Q1, ..., revenueGrowth_Q4.

        Optimized for large datasets: avoids melt/pivot_table explosion by chunking features.
        """
        print("Pivot quarterly features into wide format (optimized)")

        # Define identifiers
        id_vars = ["ticker", "year", "period"]
        value_vars = [c for c in df.columns if c not in id_vars]

        # Split feature columns into chunks to reduce memory usage
        chunks = [value_vars[i:i+chunk_size] for i in range(0, len(value_vars), chunk_size)]

        df_wide_chunks = []
        for chunk in tqdm(chunks, desc="Pivoting quarterly features"):
            # Pivot only the current chunk
            pivoted = df.pivot(index=["ticker", "year"], columns="period", values=chunk)

            # Flatten multiindex into e.g. revenueGrowth_Q1, revenueGrowth_Q2
            pivoted.columns = [f"{col}_{period}" for col, period in pivoted.columns]

            df_wide_chunks.append(pivoted)

        # Concatenate all feature chunks back together
        df_wide = pd.concat(df_wide_chunks, axis=1).reset_index()

        return df_wide


    def run_quarter(self, min_year=1990, max_year=2025):
        """
        Run extraction for quarterly data.
        """
        merged_df = None
        for i, table in tqdm(enumerate(self.input_table_names), desc="Quarterly Table progress", total=len(self.input_table_names)):
            df = self.extract_features_quarter(table, self.prefixes[i], min_year, max_year)
            self.features_quarterly[table] = df

            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on=["ticker", "year", "period"], how="outer")

        self.full_df_quarterly = merged_df.sort_values(["ticker", "year", "period"]).reset_index(drop=True)
        
        return self.full_df_quarterly

    def run(self, min_year=1990, max_year=2025, run_quarterly = True):
        """
        Wrapper method to run both FY and quarterly extraction.
        """
        fy = self.run_fy(min_year, max_year)
        if run_quarterly:
            qt = self.run_quarter(min_year, max_year)
            qt_wide = self.pivot_quarterly_features(qt)
            return fy, qt_wide
        else:
            return fy
