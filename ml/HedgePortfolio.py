from tqdm import tqdm
import pandas as pd
import pickle
import numpy as np


class HedgePortfolio:
    def __init__(
        self, 
         ml_dataset:pd.DataFrame, #dataframe created by MLDatasetBuilderCSV -->  df from which we extract the indices of the test sentences --> as all datasets share same test/train data, the choice is irrelevant and can be one of either df_fin, df_fin_mean_sentiment, df_fin_text
        base_path:str ="paper2/Results/ML/Financials", #base path where ML results are stored
        feature_set:str = "basic", #type of text features we use, e.g. basic, vamossy, renault, etc.
    ):
        self.base_path = base_path
        self.feature_set = feature_set
        self.ml_dataset = ml_dataset

    def _load_ml_results(
        self, 
        ml_config:dict, #dictionary of ML configuration --> used to construct the path to the results 
        model_name:str, # name of the ML model for which we want to load the results --> lr, rf or xgb (potentially with _detrend)
        rel_cols:list = ["ticker", "year", "target", "prob", "y_pred", "y_true"], #list with the columns we want to keep and use in further downstream methods
        df_name:str = "fin_text_transformer",  #df name for which we want to construct and evaluate the Hedge Portfolio - one of ["fin", "fin_mean_sentiment", "fin_text_transformer"] 
    ):
        """
        Helper method to load ML results
        """
        #compose prefix and suffixes based on config and method input
        prefix = f"{model_name}_swade" if ml_config.get("feature_set") == "swade" else f"{model_name}"
        suffix = "_cik_symbol" if ml_config.get("sample") == "cik_symbol" else ""
        suffix2 = "" if ml_config.get("exclude_quarter_features") else "_incl_q_feats"
            
        #compose full filename
        result_loc = (
            f"{self.base_path}/"
            f"{prefix}_{ml_config.get('target_period')}_{ml_config.get('fintextsim')}{suffix}{suffix2}.pkl"
        )
        print(f"Results will be loaded from {result_loc}")

        #load results
        with open(result_loc, "rb") as file:
            data = pickle.load(file)
        
        #extract the predicted IDs from the ML dataset
        predicted_idx = list(data["results"][self.feature_set][df_name]["X_test"].index) # identical for all dfs --> fin, fin_mean_sentiment and fin_text_transformer --> same input and same training/test indices
        print(f"Number of predictions: {len(predicted_idx)}")
        
        #extract the corresponding probabilities
        predicted_probs = data["results"][self.feature_set][df_name]["test_results"]["y_prob"].get() # identical for all dfs and transform to numpy --> fin, fin_mean_sentiment and fin_text_transformer
        print(f"Number of predictions: {len(predicted_probs)}")

        #extract the corresponding predictions
        y_pred = data["results"][self.feature_set][df_name]["test_results"]["y_pred"].get() # identical for all dfs and transform to numpy --> fin, fin_mean_sentiment and fin_text_transformer
        print(f"Number of predictions: {len(predicted_probs)}")

        #extract the corresponding predictions
        y_true = data["results"][self.feature_set][df_name]["test_results"]["y_true"].get() # identical for all dfs and transform to numpy --> fin, fin_mean_sentiment and fin_text_transformer
        print(f"Number of predictions: {len(predicted_probs)}")
        
        #extract relevant rows from ml_dataset --> those which are in testset
        df_probs = self.ml_dataset.iloc[predicted_idx].copy() #filter only for the relevant idx from ML dataset --> equal for all dfs --> we keep same rows
        df_probs["prob"] = predicted_probs #add probabilities to the df
        df_probs["y_pred"] = y_pred
        df_probs["y_true"] = y_true

        self.calculate_classification_metrics(df_probs, y_pred_name = "y_pred")
        return df_probs[rel_cols].reset_index(drop = True)


    def _prepare_portfolio_df(
        self, 
        df_probs:pd.DataFrame, #output from _load_ml_results method --> df generated from testset with probabilities, predicted and actual targets
        filing_df:pd.DataFrame, #dataframe consisting of filing dates and ends of periods of year of report for ticker, year_of_report combinations  
        months_offset:int = 3, #offset between start of portfolio and end of fiscal year --> Chen et al. 2022 use 3 months 
        months_portfolio_duration:int = 12, #offset between start and end of portfolio --> Chen et al. 2022 use 12 months 
        rel_cols:list = ["ticker", "year", "target", "prob", "y_pred", "y_true", "period_of_report", "portfolio_start", "portfolio_end"], #list of columns we want to keep in further downstream methods,
        shift_year:int = 0, #analyze performance over different years
    ):
        """
        Prepare portfolio dataframe by adding period of report from filing dates
        Create portfolio start and endpoints using month offsets
        """
        #merge df with probabilities with filing-dates df --> get end of fiscal period

        """
        optionally shift year --> we predict earning changes in target year with predictors from year -1 --> form the portfolio after the filing in t-1 and hold it right until filing for t is released
        e.g. with period_of_report 2023-12-31 and fiscal year 2023, we want to predict 2024 - we want to form portfolio at end of march 2024 (after info is available) and hold it until march 2025 (right before filing for fiscal year 2024 is released) --> subtract shift year to get results for other future years --> e.g. if shift = 1: get period_of_report 2024-12-31 for fiscal year 2023 --> portfolio start at end of march 2025 and hold until march 2026
        """
        filing_df = filing_df.copy()
        filing_df["pred_year"] = filing_df["year"] + 1 - shift_year #add 1 to the year to form prediction year: e.g. with period_of_report 2023-12-31 and fiscal year 2023, we want to predict 2024 - we want to form portfolio at end of march 2024 (after info is available) and hold it until march 2025 (right before filing for fiscal year 2024 is released) --> subtract shift year to get results for other future years --> e.g. if shift = 1: get period_of_report 2024-12-31 for fiscal year 2023 --> portfolio start at end of march 2025 and hold until march 2026
        filing_df.drop(columns = ["year"], inplace = True) #drop year column
        portfolio_df = df_probs.merge(filing_df, left_on = ["ticker", "year"], right_on = ["ticker", "pred_year"], how = "inner") #merge on pred_year --> get correct year-of-report mapping
        portfolio_df.drop(columns = ["pred_year"], inplace = True) #drop_pred_year column again
     
        
        print(f"Number of entries without period of report: {df_probs.shape[0] - portfolio_df.shape[0]}")
        portfolio_df["period_of_report"] = pd.to_datetime(portfolio_df["period_of_report"]) #ensure date format
        
        #add timing columns
        portfolio_df["portfolio_start"] = portfolio_df["period_of_report"] + pd.DateOffset(months = months_offset)
        portfolio_df["portfolio_end"] = portfolio_df["portfolio_start"] + pd.DateOffset(months = months_portfolio_duration)
        
        return portfolio_df[rel_cols]


    def _get_price_df(
        self, 
        result_dict:dict, #result dictionary containing information per ticker 
        ticker:str, #ticker for which we want to create the price df
        price_cat:str = "adjClose"
    ):
        """
        get price information per ticker
        """
        #extract stock information for ticker
        prices = result_dict[ticker]["financials"]["stock_chart_daily"]["historical"]
    
        #transform to df
        df_price = pd.DataFrame(prices)
        df_price["date"] = pd.to_datetime(df_price["date"]) #ensure date format
        df_price.sort_values("date", inplace=True) #sort values to ensure correct calculation of return
        df_price["return"] = df_price[price_cat].pct_change() #return defined as percentage change of adjusted close price
    
        return df_price.reset_index(drop = True) 

    def _get_market_cap(
        self, 
        result_dict:dict, #result dictionary containing information per ticker 
        ticker:str, #ticker for which we want to extract market cap
        year:int #fiscal year for which we want to extract market cap
    ):
        """
        Get market capitalization to apply value weighting
        """
        #check if ticker is present in result_dict
        if ticker not in result_dict:
            print(f"Ticker {ticker} not found in result_dict.")
            return np.nan

        #check if key metrics table is available for the ticker
        key_metrics = result_dict[ticker].get("financials", {}).get("key_metrics", [])
        if not key_metrics:
            print(f"No key_metrics found for {ticker}.")
            return np.nan

        #create list to store results
        rows = []
        
        #extract key metrics table
        key_metrics = result_dict[ticker]["financials"]["key_metrics"]
        #iterate over each entry/year and extract market cap
        for entry in key_metrics:
            fiscal_year = int(entry.get("calendarYear"))
            if fiscal_year == year:
                 return entry.get("marketCap")

        print(f"No marketCap found for {ticker} in year {year}.")
        return np.nan #return NA if no value can be found

    def compute_cumulative_returns(
        self, 
        portfolio_df:pd.DataFrame, #portfolio df with ticker, year and end and start of portfolio period 
        market_df_daily:pd.DataFrame, #df with daily market information 
        result_dict:dict, #dictionary containing results per ticker
        min_trading_days:int = 210, #minimum number of trading days per year to not be removed from the calculation of cumulative returns
    ):
        """
        Compute cumulative stock and market returns for all portfolio windows
        """
        rows = []
        
        #iterate over each row in portfolio df
        for idx, x in tqdm(portfolio_df.iterrows(), desc = "Calculating cumulative return", total = portfolio_df.shape[0]):    
            #extract relevant information from the row
            start = x["portfolio_start"]
            end = x["portfolio_end"]
            ticker = x["ticker"]
            year = x["year"]
            prob = x["prob"]
            target = x["target"]
            y_pred = x["y_pred"]
        
            #get price df and market cap
            price_df = self._get_price_df(result_dict, ticker) #get price df
            market_cap = self._get_market_cap(result_dict, ticker, year) #get market cap for ticker/year combination
        
            #filter stock prices for portfolio window
            window_df = price_df.loc[
                (price_df["date"] >= start) & (price_df["date"] < end),
                ["date", "adjClose", "return"]
            ]
            #merge window df with daily S&P500
            merged = window_df.merge(market_df_daily, on = "date", how = "inner", suffixes = ("_stock", "_market")) #inner join to ensure same dates
            
            #handle cases which do not fulfill the min_trading_days criteria
            num_trading_days = merged.shape[0]
            if num_trading_days < min_trading_days:
                continue

            #calculate cumulative returns of stock and market
            cumulative_return_stock = (1 + merged["return_stock"]).prod() - 1
            cumulative_return_market = (1 + merged["return_market"]).prod() - 1

            #compute excess return --> stock return - market return
            excess_return = cumulative_return_stock - cumulative_return_market
            #annualized_excess = (1 + excess_return) ** (252 / num_trading_days) - 1
            
            row = {
                "ticker": ticker,
                "year": year,
                "target": target,
                "prob": prob,
                "y_pred": y_pred,
                "portfolio_start_date": start,
                "portfolio_end_date": end,
                "num_trading_days": num_trading_days,
                "cumulative_return_stock": cumulative_return_stock,
                "cumulative_return_market": cumulative_return_market,
                "excess_return": excess_return,
                "market_cap": market_cap
            }
            rows.append(row)

        return_df = pd.DataFrame(rows)

        #print mean statistics of market and stock return
        print("\nMean cumulative market return per year: ")
        mean_cum_market_return_year = return_df.groupby(["year"])["cumulative_return_market"].mean()
        print(mean_cum_market_return_year)

        print("\nMean cumulative stock return per year: ")
        mean_cum_stock_return_year = return_df.groupby(["year"])["cumulative_return_stock"].mean()
        print(mean_cum_stock_return_year)

        print("\nMean cumulative stock return per year - value-weighted: ")
        mean_cum_stock_return_year_value_weighted = return_df.groupby("year").apply(lambda g: np.average(g["excess_return"], weights=g["market_cap"])).reset_index(name="avg_excess_return")
        print(mean_cum_stock_return_year_value_weighted)
        print("\n\n")
            
        return return_df
        

    def assign_positions(
        self, 
        return_df:pd.DataFrame, #dataframe with returns, probabilities 
        long_threshold:float=0.6, #threshold to assign long position in portfolio --> if x > threshold: long 
        short_threshold:float =0.4 #threshold to assign short position in portfolio --> if x < threshold: short
    ):
        """
        Assigns long/short positions based on predicted probabilities.
        """
        return_df = return_df.copy()

        #Assign portfolio position based on ML probabilities
        return_df["position"] = -1  # default: no position --> cases where probabilities lie between threshold
        return_df.loc[return_df["prob"] >= long_threshold, "position"] = 1   # long
        return_df.loc[return_df["prob"] <= short_threshold, "position"] = 0 # short

        #print the distribution of portfolio positions
        print(return_df.groupby("position").size())
        return return_df

    def _create_statistical_summary(
        self, 
        return_df:pd.DataFrame, #dataframe with returns 
        groupby_variable:str,  #variable for which we want create the statistical summary --> e.g. for the portfolio position, y_true or ML pred (50%)
        cat: str = "Hedge",
        value_weighted: bool = True,
        n_bootstrap=10000,
        bootstrap_by_year=False
    ):
        """
        Calculate summary and report statistical values
        Goal: evaluate whether ML model can produce economically significant predictions --> can it identify stocks that outperform the market?
        Sort stock into short long position based on predicted probabilities; compute excess return over the market; test whether difference in mean excess return between long and short portfolios is statistically significant
        """
        df = return_df.copy()

        #-----1. Filter valid observations
        #filter for binary groups --> long vs short
        df = df[df[groupby_variable].isin([1, 0])]
        if len(df[groupby_variable].unique()) <= 1:
            print("No calculation possible --> only one assigned position.")
            return None
        
        #drop missing or zero market caps when weighting
        if value_weighted:
            df = df[df["market_cap"].notna() & (df["market_cap"] > 0)]


        #--------2. Compute portfolio mean returns
         # Weighted or unweighted mean
        if value_weighted:
            #calculate overarching summary --> all years
            summary = (
                df.groupby(groupby_variable)
                  .apply(lambda g: np.average(g["excess_return"], weights=g["market_cap"]))
                  .reset_index(name="avg_excess_return")
            )
        else:
            summary = (
                df.groupby(groupby_variable)["excess_return"]
                  .mean()
                  .rename("avg_excess_return")
                  .reset_index()
            )
    
        print(f"\n{cat} summary ({'value-weighted' if value_weighted else 'equal-weighted'}):")
        print(summary)
        
        #construct hedge portfolio = long − short
        # Hedge portfolio = long − short
        if set(summary[groupby_variable]) >= {0, 1}:
            hedge_return = (
                summary.loc[summary[groupby_variable] == 1, "avg_excess_return"].iloc[0]
                - summary.loc[summary[groupby_variable] == 0, "avg_excess_return"].iloc[0]
            )
            print(f"{cat} average excess return: {hedge_return:.3%}")
        else:
            missing_groups = {0, 1} - set(summary[groupby_variable])
            print(f"Missing group(s) {missing_groups}. Cannot compute hedge return.")
            hedge_return = np.nan
        

        # --- Statistical test (overall) ---
        from scipy import stats
        #separate the two samples: excess return of long portfolio and short portfolio
        long_excess = df.loc[df[groupby_variable] == 1, "excess_return"]
        short_excess = df.loc[df[groupby_variable] == 0, "excess_return"]
        
        #Null hypothesis: no difference between returns of long and short portfolios
        if long_excess.empty or short_excess.empty:
            print("One of the portfolios is empty — skipping t-test.")
            t_stat, p_val = np.nan, np.nan
        else:
            t_stat, p_val = stats.ttest_ind(long_excess, short_excess, equal_var=False)
            print(f"{cat} t-statistic = {t_stat:.2f}, p-value = {p_val:.3f}")

        #t-statistic: how many standard errors apart are the means --> the higher, the higher the statistical value
        #p-value: probability that difference arises purely from noise/chance


        # --- Bootstrapping ---
        def _bootstrap_sample(df, groupby_variable, value_weighted):
            """
            One bootstrap iteration returning a hedge return.
            """
            if value_weighted:
                #split long and short portfolios
                long = df.loc[df[groupby_variable] == 1]
                short = df.loc[df[groupby_variable] == 0]

                #compute weighted mean for each portfolio
                long_mean = np.average(long["excess_return"], weights=long["market_cap"]) if len(long) > 0 else np.nan
                short_mean = np.average(short["excess_return"], weights=short["market_cap"]) if len(short) > 0 else np.nan
            else:
                #split long and short portfolios and comput simple mean --> not weighted by market cap
                long_mean = df.loc[df[groupby_variable] == 1, "excess_return"].mean()
                short_mean = df.loc[df[groupby_variable] == 0, "excess_return"].mean()
            return long_mean - short_mean #hedge return defined as long - short --> hedge return for this resample
    
        bootstrap_hedge_returns = []
        
        for _ in range(n_bootstrap):
            #handle boostrapping for years
            if bootstrap_by_year and "year" in df.columns:
                sampled_years = np.random.choice(df["year"].unique(), size=len(df["year"].unique()), replace=True)
                sample = df[df["year"].isin(sampled_years)]
            else:
                #bootstrapping over full sample
                sample = df.sample(frac=1.0, replace=True) #resample full df with replacement
            
            #compute bootstrap return for this bootstrap sample
            bootstrap_hedge_returns.append(_bootstrap_sample(sample, groupby_variable, value_weighted))
    
        bootstrap_hedge_returns = np.array(bootstrap_hedge_returns) #convert to numpy array for numerical operations
        ci_low, ci_high = np.percentile(bootstrap_hedge_returns, [2.5, 97.5]) #compute confidence intervals using percentiles
        boot_mean = bootstrap_hedge_returns.mean() #compute mean of all bootstrap hedge returns

        # --- Compute bootstrapped p-value ---
        # Assuming null hypothesis: hedge return = 0
        #Count how often resampled hedge returns are on the opposite side of zero relative to observed hedge return - double it for a two-sided test.
        p_value_boot = 2 * min(
            np.mean(bootstrap_hedge_returns <= 0), #counts fraction of bootstrap samples where hedge is smaller than 0 --> how often does a hedge return lower than 0 appear
            np.mean(bootstrap_hedge_returns >= 0) #counts fraction of bootstrap samples where hedge is greater than 0
        )
    
        print(f"{cat} bootstrap mean = {boot_mean:.3%}, 95% CI = [{ci_low:.3%}, {ci_high:.3%}], p_value = {p_value_boot}")

        # ---NULL RANDOMIZATION TEST ---
        """
        Randomly assign long and short labels and check whether results differ from actual hedge portfolio
        Simulation of a world where the model predictions have no predictive power
        """
        #compute number of stocks in long and short position
        n_long = (df[groupby_variable] == 1).sum()
        n_short = (df[groupby_variable] == 0).sum()

        #initialize list to store results
        null_hedge_returns = []

        #perform bootstrap sampling --> check if random portfolio can achieve similar results compared to the one constructed based on our predictions
        for _ in range(n_bootstrap):
            shuffled = df.copy() #copy the df as bas for shuffled df
            shuffled_idx = np.random.permutation(df.index) #create shuffled indices
            shuffled["rand_label"] = 0 #assign intitial short position --> label 0
            shuffled.loc[shuffled_idx[:n_long], "rand_label"] = 1 #assign long position to the first n_long instances
            shuffled.loc[shuffled_idx[n_long:n_long + n_short], "rand_label"] = 0 #assign short position to the last instances

            #construct null hypothesis hedge return for this run
            null_hedge = _bootstrap_sample(shuffled, "rand_label", value_weighted) 
            null_hedge_returns.append(null_hedge) #append results to lust
    
        null_hedge_returns = np.array(null_hedge_returns)
        p_value_null_two_sided = np.mean(np.abs(null_hedge_returns) >= abs(hedge_return)) #p-value: share of H0 random hedge return which are greater than actual hedge return --> Measures whether hedge return is unusually large in magnitude (either direction)
        p_value_null_one_sided = np.mean(null_hedge_returns >= hedge_return) #tests whether hedge return is unusually high (model beats random long/short)
    
        print(f"{cat} randomization-test p_null (two-sided) = {p_value_null_two_sided:.5f} - (does model produce unusually large hedge return in unusually large magnitude (positive and negative)")
        print(f"{cat} randomization-test p_null (one-sided) = {p_value_null_one_sided:.5f} - (does model produce unusully large heedge return --> does model beat random long/short)") 
        
    
        # --- Yearly summary ---
        yearly_results = []
        for year, group in df.groupby("year"):
            if len(group[groupby_variable].unique()) < 2:
                continue  # skip if only one position that year
    
            if value_weighted:
                yearly_summary = (
                    group.groupby(groupby_variable)
                         .apply(lambda g: np.average(g["excess_return"], weights=g["market_cap"]))
                         .reset_index(name="avg_excess_return")
                )
            else:
                yearly_summary = (
                    group.groupby(groupby_variable)["excess_return"]
                         .mean()
                         .rename("avg_excess_return")
                         .reset_index()
                )
    
            if set(yearly_summary[groupby_variable]) >= {0, 1}:
                yearly_hedge_return = (
                    yearly_summary.loc[yearly_summary[groupby_variable] == 1, "avg_excess_return"].iloc[0]
                    - yearly_summary.loc[yearly_summary[groupby_variable] == 0, "avg_excess_return"].iloc[0]
                )
            else:
                yearly_hedge_return = np.nan
    
            yearly_results.append({
                "year": year,
                "hedge_return": yearly_hedge_return
            })
    
        yearly_summary_df = pd.DataFrame(yearly_results)
        print("\nYearly hedge returns:")
        print(yearly_summary_df.round(4))
    
        results = {
            "summary": summary,
            "hedge_return": hedge_return,
            "t_stat": t_stat,
            "p_val": p_val,
            "yearly_summary": yearly_summary_df,
            "boot_mean": boot_mean,
            "boot_ci_low": ci_low,
            "boot_ci_high": ci_high,
            "p_value_boot": p_value_boot,     # from bootstrap CI
            "p_value_null_two_sided": p_value_null_two_sided,     #does model generate larger hedge return in absolute values --> two-sided
            "p_value_null_one_sided": p_value_null_one_sided, #does model generate larger hedge return - one direction only
        }
        return results
            

    def generate_statistical_summaries(
        self, 
        return_df: pd.DataFrame, #dataframe with returns
        value_weighted: bool = True,
    ):
        """
        Calculate summary and report statistical values for the different categories (hedge based on assigned positions, hedge based on the 0.5 predictions, hedge based on perfect foresight)
        """
        #average excess return by position based on probabilities
        print("\nHedge portfolio (threshold-assignments):")
        results_hedge = self._create_statistical_summary(return_df, "position", cat = "hedge - threshold", value_weighted = value_weighted)

        print("\nHedge portfolio (prediction):")
        results_hedge_pred = self._create_statistical_summary(return_df, "y_pred", cat = "hedge - pred 0.5", value_weighted = value_weighted)

        print("\nPerfect Foresight:")
        results_pf = self._create_statistical_summary(return_df, "target", cat = "Perfect foresight", value_weighted = value_weighted)
        return results_hedge, results_hedge_pred, results_pf

    def calculate_classification_metrics(
        self,
        df:pd.DataFrame, #df 
        y_true_name:str = "y_true", #name of the column which carries the true y-values
        y_pred_name:str = "position" #name of column which we want to compare against y-ture
    ):
        """
        compare classification metrics on threshold dataset
        """
        #check if results match the ones from ML prediction
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

        #filter df so that it is only binary
        df = df[df[y_pred_name].isin([0,1])]
        
        # position = model prediction (1 = predict increase, 0 = predict decrease)
        y_true = df[y_true_name]
        y_pred = df[y_pred_name]
        
        # compute metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        print("Model performance based on portfolio classification:")
        print(f"Accuracy : {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall   : {rec:.3f}")
        print(f"F1 Score : {f1:.3f}")
        print("Confusion matrix:")
        print(cm)

    def run(
        self, 
        ml_config:dict, #dictionary of ML configuration --> used to construct the path to the results
        model_name:str, # name of the ML model for which we want to load the results --> lr, rf or xgb (potentially with _detrend)
        result_dict:dict, #dictionary with results stored by ticker
        market_df_daily:pd.DataFrame, #df with daily market information 
        filing_df:pd.DataFrame, #df with filing dates/end of period of reports and fiscal year per ticker, year combination
        df_name:str = "fin_text_transformer",  #df name for which we want to construct and evaluate the Hedge Portfolio - one of ["fin", "fin_mean_sentiment", "fin_text_transformer"] 
        short_threshold:float = 0.5, #threshold to assign short position in portfolio --> if x < threshold: short
        long_threshold:float = 0.5, #threshold to assign long position in portfolio --> if x > threshold: long  
        value_weighted:bool = True,
        min_trading_days:int = 100,
        shift_year = 0
    ):
        """
        Wrapper method to run evaluation
        """
        #1. Generate the base df with ticker, year, target and probability
        df_probs = self._load_ml_results(ml_config, model_name, df_name = df_name)

        #2. Load filing dates and create portfolio df
        portfolio_df = self._prepare_portfolio_df(df_probs, filing_df, shift_year = shift_year)

        #3. compute return df
        return_df = self.compute_cumulative_returns(portfolio_df, market_df_daily, result_dict, min_trading_days = min_trading_days)

        #4. Add long/short positions
        return_df_ls = self.assign_positions(return_df, long_threshold, short_threshold)

        #5. Compute results
        results_hedge, results_hedge_pred, results_pf = self.generate_statistical_summaries(return_df_ls, value_weighted = value_weighted)

        return results_hedge, results_hedge_pred, results_pf

    def run_cumulative_return_base(
        self, 
        ml_config:dict, #dictionary of ML configuration --> used to construct the path to the results
        model_name: str, #name of the ML model for which we want to load the results --> lr, rf or xgb (potentially with _detrend)
        result_dict: dict, #dictionary with results stored by ticker
        market_df_daily: pd.DataFrame, #df with daily market information 
        filing_df:pd.DataFrame, #df with filing dates/end of period of reports and fiscal year per ticker, year combination
        df_name:str = "fin_text_transformer",  #df name for which we want to construct and evaluate the Hedge Portfolio - one of ["fin", "fin_mean_sentiment", "fin_text_transformer"] 
        min_trading_days:int = 100,
        shift_year:int = 0
    ):
        """
        Wrapper method to run creation of cumulative return df --> identical for all ML models and datasets as we have the same test data
        """
        #1. Generate the base df with ticker, year, target and probability
        df_probs = self._load_ml_results(ml_config, model_name, df_name = df_name)
        self.calculate_classification_metrics(df_probs, y_pred_name = "y_pred")

        #2. Load filing dates and create portfolio df
        portfolio_df = self._prepare_portfolio_df(df_probs, filing_df, shift_year = shift_year)

        #3. compute return df
        return_df = self.compute_cumulative_returns(portfolio_df, market_df_daily, result_dict, min_trading_days = min_trading_days)

        return return_df

    def load_model_probs_and_run_eval(
        self, 
        return_df:pd.DataFrame, # df created by run_cumulative_return_base
        ml_config:dict, #dictionary of ML configuration --> used to construct the path to the results 
        model_name: str, #name of the ML model for which we want to load the results --> lr, rf or xgb (potentially with _detrend)
        long_threshold:float = 0.5, #threshold to assign long position in portfolio --> if x > threshold: long  
        short_threshold:float = 0.5, #threshold to assign short position in portfolio --> if x < threshold: short
        df_name:str = "fin_text_transformer",  #df name for which we want to construct and evaluate the Hedge Portfolio - one of ["fin", "fin_mean_sentiment", "fin_text_transformer"] 
        value_weighted:bool = True,
    ):
        """
        Load model specific probabilities and assign positions
        """
        #1. Generate the base df with ticker, year, target and probability based on the desired model
        df_probs = self._load_ml_results(ml_config, model_name, df_name = df_name)

        #2. Overwrite probabilities and y_pred from base model with the ones from the chosen model
        return_df.drop(columns = ["prob", "y_pred", "y_true", "target"], inplace = True, errors = "ignore")
        return_df = return_df.merge(df_probs, on = ["ticker", "year"], how = "left")
        
        #3. Add long/short positions
        return_df_ls = self.assign_positions(return_df, long_threshold, short_threshold)
        self.calculate_classification_metrics(return_df_ls, y_pred_name = "position")

        #4. Compute results
        results_hedge, results_hedge_pred, results_pf = self.generate_statistical_summaries(return_df_ls, value_weighted = value_weighted)
        return results_hedge, results_hedge_pred, results_pf