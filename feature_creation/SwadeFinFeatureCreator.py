import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm



class SwadeFinFeatureCreator:
    """
    Idea: create the base for extracting the 15 features used by Swade et al. 2023 and Koval et al. 2024 - including other features can introduce noise
    """
    def __init__(self):
        """
        Test
        """


    def extract_rows(self,
        result_dict:dict, #result dict
        result_cat:str = "financials",
        input_table_name:str = "stock_chart_daily", #key from which we want to extract variables
        fields_to_extract = ["adjClose", "volume", "changePercent"]
        ):
        """
        Extract entries from the JSON input and transform them to rows followed by transformation into a dataframe
        date: boolean: if True, the date is in %Y-%m-%d format. Switch to False if the year is given directly
        """
        rows = []
    
        #iterate over each ticker
        for ticker, data in tqdm(result_dict.items(), desc="Progress"):
            result_dict
            rel_data = data.get(result_cat, []).get(input_table_name, []) #extract the relevant input table

            #check if "historical" exists
            if not rel_data or "historical" not in rel_data:
                #skip this ticker if no historical data
                continue
    
            for entry in rel_data["historical"]: #iterate over each entry in the relevant input table
                date = entry.get("date")
                
                row = {
                    "ticker": ticker,
                    "date": date,
                }
                for field in fields_to_extract:
                    row[field] = entry.get(field, np.nan) #extract the relevant fields
    
                rows.append(row) #append the row with relevant fields and id_vars to the rows
    
        #create df from the rows
        df = pd.DataFrame(rows)
        
        #drop duplicate rows of ticker, year --> before sorting to ensure that newest values are kept  
        df = df.drop_duplicates(subset = ["ticker", "date"])
    
        #sort values for calculating YoY percentage changes
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        return df

    def _winsorize_stock_df(self, stock_df, percentile_low = 0.01, percentile_high = 0.99):
        """
        Winsorize the stock df by removing zero or invalid prices as well as zero volume and clipping of data based on percentiles
        PROBLEM: will confuse the event windows --> dates are missing due to winsorization; 
        SOLUTION: do not apply the winsorization - filter/winsorize/remove either in target df or in ML dataset used for prediction
        """
        #filter invalid prices/volumes
        df = stock_df.copy()
        df = df[df["adjClose"] > 0]       #remove zero or invalid prices
        df = df[df["volume"] > 0]         #remove zero volume

        #sort by ticker and date - calculate daily return
        df = df.sort_values(["ticker", "date"])
        df["daily_return"] = df.groupby("ticker")["adjClose"].pct_change()
        
        #compute low and high percentile from valid data
        lower, upper = df["daily_return"].quantile([percentile_low, percentile_high])
        
        print(f"Clipping daily returns to between {lower:.3f} and {upper:.3f} - values between {percentile_low} and {percentile_high} percentile are kept.")
        
        #clip extreme values
        df["daily_return"] = df["daily_return"].clip(lower=lower, upper=upper)
        
        #drop NaN from the first day of each ticker
        df = df.dropna(subset=["daily_return"]).reset_index(drop = True)

        return df

    def extract_event_windows(self,
                              stock_df, 
                              filing_df, 
                              pre_days:int = 252, #full year back
                              post_days:int = -int(252/12) #exclude the last month
                              ):
        """
        function to extract relevant information from stock df based on pre_days (trading days) before filing date and post_days (trading days) past filing date

        medium-term price momentum: past year (252 trading days) up until one month before filing date (252/12) trading days --> 252 pre_days, -252/12 post_days
        short-term price reversal: past month before filing date (252/12) pre_days, 0 post_days
        """
        #convert date to datetime
        stock_df['date'] = pd.to_datetime(stock_df['date'])
        filing_df['filing_date'] = pd.to_datetime(filing_df['filing_date'])
    
        #sort df
        stock_df = stock_df.sort_values(['ticker', 'date']).copy()
    
        #create dictionary for faster lookup per ticker
        stock_by_ticker = {ticker: df for ticker, df in stock_df.groupby('ticker')}
    
        #merge dataframes to only iterate over filing dates which are also in stock info
        filing_dates_merge = filing_df.merge(stock_df, left_on = ["ticker", "filing_date"], right_on = ["ticker", "date"], how = "inner")
        filing_dates_merge = filing_dates_merge[[col for col in filing_dates_merge.columns if col in filing_df.columns]] #ensure that we maintain only the original columns
    
        #create list to store results
        all_windows = []
    
        #iterate over each row in merged df
        for _, row in tqdm(filing_dates_merge.iterrows(), total = filing_dates_merge.shape[0], desc="Extracting event windows"):
            ticker = row['ticker']
            filing_date = row['filing_date']
            year = row['year'] #year from period of report originating from employee - count - filing date API
    
            if ticker not in stock_by_ticker:
                continue
    
            #extract the data relevant for ticker from dictionary
            ticker_data = stock_by_ticker[ticker]
            ticker_data = ticker_data.reset_index(drop=True)
    
            #find index of filing_date in trading days
            date_idx = ticker_data[ticker_data['date'] == filing_date].index
            if len(date_idx) == 0:
                continue
            idx = date_idx[0]
    
            #define start and end idx
            start_idx = max(0, idx - pre_days) #handle cases where there are less than pre_days observations before filing date
            end_idx = min(len(ticker_data), idx + post_days + 1) #handle cases where there are less than post_days + 1 observations after filing date
    
            window_df = ticker_data.iloc[start_idx:end_idx].copy() #extract the relevant data from full set
            window_df['event_filing_date'] = filing_date #add the filing date to df
            window_df['event_ticker'] = ticker #add ticker to df
            window_df['year'] = year #add the filing date to df
            
            
            all_windows.append(window_df)
    
        event_window_df = pd.concat(all_windows, ignore_index=True)
    
        return event_window_df
    
    
    
    def generate_window_metrics(
        self, 
        stock_event_window, 
        stock_df
    ):
        """
        Generate metrics based on the window --> check closing price at pre_days and post_days
        """
        #calculate prediction (smallest date) and target (largest date) per filing date
        window_start_dates = stock_event_window.groupby(["ticker", "event_filing_date"])["date"].min().reset_index(name="window_start")
        window_end_dates = stock_event_window.groupby(["ticker", "event_filing_date"])["date"].max().reset_index(name="window_end")
        
        #merge to get a single dataframe with both dates
        date_pairs = window_start_dates.merge(window_end_dates, on=["ticker", "event_filing_date"])
        
        #melt the date_pairs to have a single column for merging with stock_df
        date_pairs_melted = pd.melt(date_pairs, 
                                    id_vars=["ticker", "event_filing_date"], 
                                    value_vars=["window_start", "window_end"],
                                    var_name="date_type", value_name="date")
        
        #merge to get stock values for both dates
        date_values = date_pairs_melted.merge(stock_df, on=["ticker", "date"], how="left")
        
        #pivot values so that we have separate columns for prediction and target values
        date_values_pivot = date_values.pivot(index=["ticker", "event_filing_date"], 
                                               columns="date_type", 
                                               values=["adjClose", "volume"]).reset_index()
        
        #flatten column names
        date_values_pivot.columns = [f"{j}_{i}" if j else i for i, j in date_values_pivot.columns]

        #date_values_pivot = date_values_pivot[["ticker", "event_filing_date", ]]
        
        return date_values_pivot

    def run_stock_features(
        self, 
        stock_df, 
        filing_df
    ):
        """
        Wrapper method to extract short- and midterm windows based on filing dates and create metrics within the window
        """
        event_window_shortterm = self.extract_event_windows(stock_df, filing_df, pre_days = int(252/12), post_days = 0)
        event_window_midterm = self.extract_event_windows(stock_df, filing_df, pre_days = 252, post_days = - int(252/12))
        event_window_full_year = self.extract_event_windows(stock_df, filing_df, pre_days = 252, post_days = 0)

        shortterm_metrics = self.generate_window_metrics(event_window_shortterm, stock_df)
        denom = shortterm_metrics["window_start_adjClose"].replace(0, np.nan) #replace 0's with NA to avoid division by 0 --> start price of 0 does not make sense --> NA
        shortterm_metrics["short_term_price_reversal"] = (shortterm_metrics["window_end_adjClose"] / denom) - 1

        midterm_metrics = self.generate_window_metrics(event_window_midterm, stock_df)
        denom = midterm_metrics["window_start_adjClose"].replace(0, np.nan)
        midterm_metrics["medium_term_price_momentum"] = (midterm_metrics["window_end_adjClose"] / denom) - 1

        full_year_metrics = self.generate_window_metrics(event_window_full_year, stock_df)
        denom = full_year_metrics["window_start_adjClose"].replace(0, np.nan)
        full_year_metrics["medium_term_price_momentum"] = (full_year_metrics["window_end_adjClose"] / denom) - 1
        


        #calculate volatility - midterm and defined as variance coefficient of adjusted close
        agg_functions = {
            "adjClose": ["var","mean"]
        }
        
        volatility = event_window_midterm.groupby(["ticker", "event_filing_date"]).agg(agg_functions).reset_index()
        # Flatten the MultiIndex columns after aggregation
        volatility.columns = ["ticker", "event_filing_date", "var", "mean"]
        
        # Calculate coefficient of variation (variance / mean)
        volatility["volatility"] = volatility["var"] / volatility["mean"] #define volatility as the variance coefficient
        volatility = volatility.drop(columns = ["var", "mean"])

        agg_functions = {
            "volume": "mean"
        }

        #calculate average number of shares traded --> base for share turnover
        share_volume = event_window_full_year.groupby(["ticker", "event_filing_date"]).agg(agg_functions).reset_index()
        share_volume.columns = ["ticker", "event_filing_date", "mean_volume"]

        #combine the features
        stock_features = midterm_metrics[["ticker", "event_filing_date", "medium_term_price_momentum"]].merge(shortterm_metrics[["ticker", "event_filing_date", "short_term_price_reversal"]],  on = ["ticker", "event_filing_date"]).merge(volatility, on = ["ticker", "event_filing_date"]).merge(share_volume, on = ["ticker", "event_filing_date"])

        #bring year from filings in
        stock_features = stock_features.merge(filing_df, left_on = ["ticker", "event_filing_date"], right_on = ["ticker", "filing_date"], how = "left") 
        stock_features = stock_features[["ticker", "year", "medium_term_price_momentum", "short_term_price_reversal", "volatility", "mean_volume"]]
        
        return stock_features