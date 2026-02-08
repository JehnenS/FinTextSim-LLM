import pandas as pd
from tqdm import tqdm

class EconomicFeatureCreator:
    def __init__(self, result_dict):
        self.result_dict = result_dict
        
    def extract_features(self):
        """
        Extract economic indicators per date.
        """
        rows = []

        #iterate over each indicator
        for indicator, table in tqdm(self.result_dict.items(), desc="Extracting economic indicators"):
            for entry in table:
                date_str = entry.get("date") #extract date and value for each entry per indicator
                value = entry.get("value")
                if date_str is None or value is None: #skip if there is no date or value
                    continue
                rows.append({"indicator": indicator, "date": date_str, "value": value})

        #transform to df
        econ_df = pd.DataFrame(rows)
        econ_df["date"] = pd.to_datetime(econ_df["date"]) #ensure date format
        econ_df = econ_df.pivot(index="date", columns="indicator", values="value").reset_index() #pivot into wide format
        econ_df = econ_df.ffill().reset_index(drop=True) #forward fill empty values per indicator
        return econ_df
    
    def add_filing_windows(self, filing_dates):
        """
        Prepares start/end windows for each filing.
        """
        #ensure date format
        filing_dates = filing_dates.copy()
        filing_dates["end_fy"] = pd.to_datetime(filing_dates["period_of_report"]) #ensure date format
        filing_dates["period_of_report"] = pd.to_datetime(filing_dates["period_of_report"]) # ensure date format
        filing_dates["filing_date"] = pd.to_datetime(filing_dates["filing_date"])

        #generate start of fiscal year and previous filing date columns by shifting by one year
        filing_dates["start_fy"] = pd.to_datetime(filing_dates.groupby("ticker")["period_of_report"].shift(1)) #shift back by one period to get start of FY
        filing_dates["prev_filing_date"] = pd.to_datetime(filing_dates.groupby("ticker")["filing_date"].shift(1)) #shift one period back to get the previous filing date
        
        #impute missing with -1 year offset
        mask_fy = filing_dates["start_fy"].isna()
        filing_dates.loc[mask_fy, "start_fy"] = filing_dates.loc[mask_fy, "period_of_report"] - pd.DateOffset(years=1)
        mask_fd = filing_dates["prev_filing_date"].isna()
        filing_dates.loc[mask_fd, "prev_filing_date"] = filing_dates.loc[mask_fd, "filing_date"] - pd.DateOffset(years=1)

        return filing_dates
    
    def compute_last_values(self, econ_df, filing_dates, date_name="FY"):
        """
        Get last observed macro values before filing/end date (via merge_asof).
        """
        #sort values to ensure correct extraction
        econ_df = econ_df.sort_values("date")

        #handle cases where we handle either FY or filing dates
        if date_name == "FY":
            key_dates = filing_dates[["ticker", "year", "end_fy"]].rename(columns={"end_fy": "key_date"})
        else:
            key_dates = filing_dates[["ticker", "year", "filing_date"]].rename(columns={"filing_date": "key_date"})
        key_dates = key_dates.sort_values("key_date")

        #merge last date --> get last value for each indicator for the last date in timeframe of FY/filing period
        econ_last = pd.merge_asof(
            key_dates, 
            econ_df, 
            left_on="key_date", 
            right_on="date", 
            direction="backward"
        ).drop(columns="date")

        econ_last = econ_last.rename(columns={col: f"{col}_last" for col in econ_df.columns if col != "date"})
        return econ_last
    
    def compute_mean_values(self, econ_df, filing_dates, date_name="FY"):
        """
        Compute mean macro values inside each fiscal/filing window.
        """
        mean_rows = []

        #iterate over filing dates
        for _, row in filing_dates.iterrows():
            #assign the correct start and end date based on choice of fiscal year or filing period
            if date_name == "FY":
                start, end = row["start_fy"], row["end_fy"]
            else:
                start, end = row["prev_filing_date"], row["filing_date"]

            mask = (econ_df["date"] >= start) & (econ_df["date"] <= end) #create a mask for the relevant dates
            econ_window = econ_df.loc[mask].mean(numeric_only=True).to_dict() #filter the df based on mask
            econ_window["ticker"] = row["ticker"]
            econ_window["year"] = row["year"]
            mean_rows.append(econ_window)

        econ_mean = pd.DataFrame(mean_rows)
        econ_mean = econ_mean.rename(columns={col: f"{col}_mean" for col in econ_df.columns if col != "date"})
        return econ_mean

    def aggregate_macro_per_ticker_year(self, econ_last, econ_mean):
        """
        Combine last + mean and compute YoY growth.
        """
        agg = econ_last.merge(econ_mean, on=["ticker", "year"], how="left")
        econ_cols = list(self.result_dict.keys())

        for col in econ_cols:
            last_col = f"{col}_last"
            growth_col = f"{col}_growth_pct"
            if last_col in agg.columns:
                agg[growth_col] = agg.groupby("ticker")[last_col].pct_change() * 100 #calculate percentage change based on last values
        return agg

    def run(self, filing_dates, date_name="FY"):
        """
        Wrapper pipeline.
        """
        econ_df = self.extract_features()
        filing_dates = self.add_filing_windows(filing_dates)
        econ_last = self.compute_last_values(econ_df, filing_dates, date_name)
        econ_mean = self.compute_mean_values(econ_df, filing_dates, date_name)
        agg_df = self.aggregate_macro_per_ticker_year(econ_last, econ_mean)
        return agg_df