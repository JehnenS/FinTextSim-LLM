import pandas as pd
import numpy as np
from tqdm import tqdm


#----------------NEW
import pandas as pd
import numpy as np
from tqdm import tqdm

class MLDatasetBuilder:
    """
    Class to build ML dataset based on CSV predictors and target df
    """
    def __init__(self, 
                 target_df: pd.DataFrame, # dataframe with year, ticker and target
                 financials: pd.DataFrame, # dataframe where financial features are stored
                 text_features_transformer: pd.DataFrame = None, # transformer-based sentence-level aggregated text features --> ticker and year_of_report as identifier
                 text_features_llm: pd.DataFrame = None #rel for P3
                ):
        
        self.target_df = target_df
        self.financials = financials
        self.text_features_transformer = text_features_transformer
        self.text_features_llm = text_features_llm


    def build_target_base(self):
        """
        Build the base for building the single dataframes --> grounded on the same observations

        CHANGE:
        The base now aligns each *target year t+1* with available *predictor-year t* 
        (e.g., 2024 EPS change is predicted using 2023 financials/text data).
        """
        #Step 1: Start base with all target rows
        base = self.target_df[["ticker", "year"]].copy()

        #shift one year backward for alignment (predictors come from t, target from t+1)
        base["pred_year"] = base["year"] - 1

        #Step 2: Restrict base if text features exist --> only include cases with available text data
        if self.text_features_transformer is not None:
            base = base.merge(
                self.text_features_transformer[["ticker", "year_of_report"]],
                left_on=["ticker", "pred_year"],   #CHANGE: match target t+1 with text year t
                right_on=["ticker", "year_of_report"], 
                how="inner"
            )
            base = base[["ticker", "year", "pred_year"]]

        elif self.text_features_llm is not None:
            base = base.merge(
                self.text_features_llm[["ticker", "year_of_report"]],
                left_on=["ticker", "pred_year"],   # CHANGE: match target t+1 with text year t
                right_on=["ticker", "year_of_report"], 
                how="inner"
            )
            base = base[["ticker", "year", "pred_year"]]

        # Step 3: Restrict target to base
        target_base = self.target_df.merge(base, on=["ticker", "year"], how="inner")

        print(f"Target base shape: {target_base.shape}")
        return target_base


    def build_all(self):
        """
        Build consistent ML datasets:
            - Financials only
            - Financials + mean_sentiment (Transformer text)
            - Financials + Transformer text features
            - Financials + LLM text features

        CHANGE:
        Merges now align predictors (financials/text) from year t 
        with the target from year t+1.
        """
        # Step 1: Create target base
        target_base = self.build_target_base()

        # Step 2: Financials only df --> merge base with financials on predictor year
        df_fin_only = target_base.merge(
            self.financials.rename(columns={"year": "pred_year"}),
            on=["ticker", "pred_year"],
            how="inner"
        )

        # Step 3: Financials + Transformer text
        df_fin_text = None
        df_text_only = None
        if self.text_features_transformer is not None:
            df_fin_text = (
                target_base.merge(
                    self.financials.rename(columns={"year": "pred_year"}), #merge on prediction year --> prediction year = year of report of the 10-K to predict t+1
                    on=["ticker", "pred_year"],
                    how="inner"
                )
                .merge(
                    self.text_features_transformer.rename(columns={"year_of_report": "pred_year"}),
                    on=["ticker", "pred_year"],
                    how="left"
                )
            )
            df_fin_text.drop(columns=["doc_id"], errors="ignore", inplace=True)
            #create text-only df
            #df_text_only = df_fin_text[['ticker', 'year', 'target', 'pred_year', 'mean_sentiment', 'mean_sentiment_topic_10', 'mean_sentiment_topic_0', 'mean_sentiment_topic_3', 'mean_sentiment_topic_11', 'mean_sentiment_topic_5', 'mean_sentiment_topic_1',       'mean_sentiment_topic_2', 'mean_sentiment_topic_9', 'mean_sentiment_topic_4', 'mean_sentiment_topic_6', 'mean_sentiment_topic_8', 'mean_sentiment_topic_7']]

        # Step 4: Financials + LLM text
        df_fin_llm = None
        if self.text_features_llm is not None:
            df_fin_llm = (
                target_base.merge(
                    self.financials.rename(columns={"year": "pred_year"}),
                    on=["ticker", "pred_year"],
                    how="inner"
                )
                .merge(
                    self.text_features_llm.rename(columns={"year_of_report": "pred_year"}),
                    on=["ticker", "pred_year"],
                    how="left"
                )
            )

        # Step 5: Financials + mean_sentiment
        df_fin_mean_sentiment = None
        if (
            self.text_features_transformer is not None 
            and "mean_sentiment" in self.text_features_transformer.columns
        ):
            df_fin_mean_sentiment = (
                target_base.merge(
                    self.financials.rename(columns={"year": "pred_year"}),
                    on=["ticker", "pred_year"],
                    how="inner"
                )
                .merge(
                    self.text_features_transformer[["ticker", "year_of_report", "mean_sentiment"]]
                        .rename(columns={"year_of_report": "pred_year"}),
                    on=["ticker", "pred_year"],
                    how="left"
                )
            )

        # Step 6: Print dataset shapes
        print(f"df_fin_only: {df_fin_only.shape}")
        if df_fin_mean_sentiment is not None:
            print(f"df_fin_mean_sentiment: {df_fin_mean_sentiment.shape}")
        if df_fin_text is not None:
            print(f"df_fin_text: {df_fin_text.shape}")
            #print(f"df_text_only: {df_text_only.shape}")
        if df_fin_llm is not None:
            print(f"df_fin_llm: {df_fin_llm.shape}")

        return df_fin_only, df_fin_mean_sentiment, df_fin_text, df_fin_llm#, df_text_only


class MLDatasetBuilder_old:
    """
    Class to build ML dataset based on CSV predictors and target df
    """
    def __init__(self, 
                 target_df: pd.DataFrame, #dataframe with year, ticker and target
                 financials: pd.DataFrame, #Dataframe where financial features are stored
                 text_features_transformer: pd.DataFrame = None, #transformer-based sentence-level aggregated text features --> ticker and year_of_report as identifier
                 text_features_llm: pd.DataFrame = None
                ):
        
        self.target_df = target_df
        self.financials = financials
        self.text_features_transformer = text_features_transformer
        self.text_features_llm = text_features_llm

    def build_target_base(self):
        """
        Build the base for building the single dataframes --> grounded on the same observations
        """
        #Step 1: Start base with all target rows
        base = self.target_df[["ticker", "year"]].copy()
    
        #Step 2: Restrict base if text features exist --> we only want to look at those cases for which we have text data
        if self.text_features_transformer is not None:
            base = base.merge(
                self.text_features_transformer[["ticker", "year_of_report"]],
                left_on=["ticker", "year"], 
                right_on=["ticker", "year_of_report"], 
                how="inner"
            )
            base = base[["ticker", "year"]]
    
        elif self.text_features_llm is not None:
            base = base.merge(
                self.text_features_llm[["ticker", "year_of_report"]],
                left_on=["ticker", "year"], 
                right_on=["ticker", "year_of_report"], 
                how="inner"
            )
            base = base[["ticker", "year"]]
    
        #Step 3: Restrict target to base
        target_base = self.target_df.merge(base, on=["ticker", "year"], how="inner")
        return target_base


    def build_all(self):
        """
        Build consistent ML datasets:
            - Financials only
            - Financials + mean_sentiment (Transformer text)
            - Financials + Transformer text features
            - Financials + LLM text features

        Shift year by 1 --> in targets, we shifted year by -1 to merge predictors with target (e.g. FY Result 2024 merged with 2023 predictors); now, we need to shift the one year back (this way, we have the target year 2024 back again instead of 2023)
        """
        #1. create target base
        target_base = self.build_target_base()

    
        #2. Financials only df --> merge base with financials on ticker and year
        df_fin_only = target_base.merge(self.financials, on=["ticker", "year"], how="inner")
        
        
        #3. Financials + Transformer text
        df_fin_text = None
        if self.text_features_transformer is not None:
            df_fin_text = df_fin_only.merge(
                self.text_features_transformer.rename(columns={"year_of_report": "year"}),
                on=["ticker", "year"],
                how="left"
            )
            
            df_fin_text.drop(columns = ["doc_id"], inplace = True) #drop the doc_id column
    
        #4. Financials + LLM text
        df_fin_llm = None
        if self.text_features_llm is not None:
            df_fin_llm = df_fin_only.merge(
                self.text_features_llm.rename(columns={"year_of_report": "year"}),
                on=["ticker", "year"],
                how="left"
            )
            
        #5. Financials + mean_sentiment
        df_fin_mean_sentiment = None
        if df_fin_text is not None and "mean_sentiment" in df_fin_text.columns:
            df_fin_mean_sentiment = df_fin_only.merge(
                df_fin_text[["ticker", "year", "mean_sentiment"]],
                on=["ticker", "year"],
                how="left"
            )

        
        #print dataset shapes
        print(f"df_fin_only: {df_fin_only.shape}")
        if df_fin_mean_sentiment is not None:
            print(f"df_fin_mean_sentiment: {df_fin_mean_sentiment.shape}")
        if df_fin_text is not None:
            print(f"df_fin_text: {df_fin_text.shape}")
        if df_fin_llm is not None:
            print(f"df_fin_llm: {df_fin_llm.shape}")
    
        return df_fin_only, df_fin_mean_sentiment, df_fin_text, df_fin_llm


