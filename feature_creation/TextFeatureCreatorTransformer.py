import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm

class TextFeatureCreatorTransformer:
    def __init__(self, sentiment_preds, topics, metadata, cik_ticker_mapping):
        self.sentiment_preds = sentiment_preds
        self.topics = topics
        self.metadata = metadata
        self.cik_ticker_mapping = cik_ticker_mapping
        self.df = None  #store intermediate dataframe
        self.features = None  #final output
        

    def build_dataframe(self):
        """
        Convert raw predictions and metadata into a DataFrame.
        """
        rows = []
        doc_ids = [entry.get("doc_id") for entry in self.metadata]

        df = pd.DataFrame({
            "doc_id": [m.get("doc_id") for m in self.metadata],
            "sentiment_class": [s[0] for s in self.sentiment_preds],
            "sentiment_prob": [s[1] for s in self.sentiment_preds],
            "topic": self.topics
        })
        
        print(f"Shape of prediction/metadata df: {df.shape}")

        self.df = df

    def get_features(self): 
        return self.features


    def compute_aggregates(self, filter_neutral_scores:bool = False):
        """
        Compute document-level and topic-level features.
        filter_neutral_scores: boolean to decide if we want to consider neutral scores --> most scores are neutral. Thus, the scores are dragged towards 1, weakening the signal
        """
        if self.df is None:
            raise ValueError("DataFrame not built. Call build_dataframe() first.")

        results = []

        df = self.df.copy()
        if filter_neutral_scores:
            df = df[df["sentiment_class"] != 1].reset_index(drop = True)

        for doc_id, group in df.groupby("doc_id"):
            result = {"doc_id": doc_id}
            result["mean_sentiment"] = group["sentiment_class"].mean()

            for topic in group["topic"].unique():
                #if topic == -1:
                   # continue
                topic_group = group[group["topic"] == topic]
                result[f"mean_sentiment_topic_{topic}"] = topic_group["sentiment_class"].mean()

            results.append(result)

        self.features = pd.DataFrame(results)
        return self

    def _group_df_(self, filter_neutral_scores:bool = False, compute_shares:bool = True, eps = 1e-5):
        """
        Helper method to group dataframe for creating features based on absolute counts for overall document and per topic per document
        """
        if self.df is None:
            raise ValueError("DataFrame not built. Call build_dataframe() first.")

        df = self.df.copy()
        #filter neutral scores if desired
        if filter_neutral_scores:
            df = df[df["sentiment_class"] != 1].reset_index(drop = True)

        #------------Overall (doc-id)
        grouped = (
            df.groupby(["doc_id", "sentiment_class"])
                .size()
                .unstack(fill_value = 0)
            .rename(columns = {0: "neg", 1: "neu", 2: "pos"})
        )
        
        grouped["num_sentences"] = grouped.sum(axis = 1) #take rowsums of all sentences to get the number of sentences
        grouped = grouped.reset_index() #flatten back into single header
        grouped.columns.name = None #avoid index to be called sentiment_class
        
        if compute_shares: #compute positive and negative shares if desired
            grouped["share_pos"] = grouped["pos"]/grouped["num_sentences"] #calculate positive and negative share - omit neutral values which weaken the signal
            grouped["share_neg"] = grouped["neg"]/grouped["num_sentences"]

        #------------topic-wise
        grouped_topic = (
            df.groupby(["doc_id", "topic", "sentiment_class"])
                .size()
                .unstack(fill_value = 0)
            .rename(columns = {0: "neg", 1: "neu", 2: "pos"})
        )
        
        grouped_topic.columns.name = None
        grouped_topic = grouped_topic.reset_index()
        
        grouped_topic["num_sentences"] = grouped_topic.sum(axis = 1) #take rowsums of all sentences to get the number of sentences

        if compute_shares:
            grouped_topic["share_pos"] = grouped_topic["pos"] / (grouped_topic["num_sentences"] + eps) #calculate positive and negative share - omit neutral values which weaken the signal
            grouped_topic["share_neg"] = grouped_topic["neg"] / (grouped_topic["num_sentences"] + eps)
        

        return grouped, grouped_topic

    def _pivot_topic_(self, grouped_topic, value_col = "mean_sentiment"):
        """
        Helper method to pivot topic df into wide format
        """
        #1. Pivot into wide format
        grouped_topic = grouped_topic.pivot(index = "doc_id", columns = "topic", values = value_col).reset_index() #pivot into wide format
        
        #2. Rename topic columns (not doc_id)
        topic_cols = [col for col in grouped_topic.columns if col != "doc_id"]
        
        grouped_topic = grouped_topic.rename(
            columns={col: f"mean_sentiment_topic_{col}" for col in topic_cols}
        )
        
        grouped_topic.columns.name = None
        return grouped_topic
        
    
    def compute_aggregates_azimi(self, eps = 1e-5, filter_neutral_scores:bool = False):
        """
        Compute document-level and topic-level features based on Azimi et al. 2021 --> Share of negative and share of positive sentences --> but, we put it into one measure by using the ratio of positive to negative
        Similar to Gupta et al. 2022
        filter_neutral_scores: boolean to decide if we want to consider neutral scores --> most scores are neutral. Thus, the scores are dragged towards 1, weakening the signal
        """
        #1.-------------Overall mean sentiment
        #1. get grouped df by document and per document-topic
        grouped, grouped_topic = self._group_df_(filter_neutral_scores = filter_neutral_scores, compute_shares = True)
        
        #2. Calculate mean sentiment following the definition by Azimi2021
        grouped["mean_sentiment"] = grouped["share_pos"] / (grouped["share_neg"] + eps) #calculate sentiment ratio by dividing share of positive sentences by share of negative sentences
        
        #3. keep only relevant columns
        grouped = grouped[["doc_id", "mean_sentiment"]] #keep only doc_id (identifier) and mean_sentiment (value)


        #2.-----------------------Topic sentiment
        #1. Calculate mean sentiment following the definition of Azimi 2021
        grouped_topic["mean_sentiment"] = grouped_topic["share_pos"] / (grouped_topic["share_neg"] + eps) #calculate sentiment ratio by dividing share of positive sentences by share of negative sentences

        #2. Pivot into wide format and rename columns
        grouped_topic = self._pivot_topic_(grouped_topic)
  
        #3. -----join mean sentiment and topic-sentiment in one df
        sentiment_scores = grouped.merge(grouped_topic, on = "doc_id")
        self.features = sentiment_scores
        return self

    def compute_aggregates_huang(self, eps = 1e-5, filter_neutral_scores:bool = False):
        """
        Compute document-level and topic-level features based on Huang et al. 2023 --> label based on highest probability (negative, neutral, positive). Then share of positive - share of negative
        filter_neutral_scores: boolean to decide if we want to consider neutral scores --> most scores are neutral. Thus, the scores are dragged towards 1, weakening the signal
        """
        #1.-------------Overall mean sentiment
        #1. get grouped df by document and per document-topic
        grouped, grouped_topic = self._group_df_(filter_neutral_scores = filter_neutral_scores, compute_shares = True)
        
        #2. Calculate mean sentiment based on definition in Huang 2023
        grouped["mean_sentiment"] = grouped["share_pos"] - grouped["share_neg"]  #calculate mean sentiment by substracting share of negatives by share of positives ratio by dividing share of positive sentences by share of negative sentences

        #3. keep only relevant columns
        grouped = grouped[["doc_id", "mean_sentiment"]] #keep only doc_id (identifier) and mean_sentiment (value)

        #2.-----------------------Topic sentiment
        #1. Calculate mean sentiment following the definition of Huang 2023
        grouped_topic["mean_sentiment"] = grouped_topic["share_pos"] - grouped_topic["share_neg"] 

        #2. Pivot into wide format and rename columns
        grouped_topic = self._pivot_topic_(grouped_topic)

        #4.-----join mean sentiment and topic-sentiment in one df
        sentiment_scores = grouped.merge(grouped_topic, on = "doc_id")
        self.features = sentiment_scores
        return self

    def compute_aggregates_zhang(self, eps = 1e-5, filter_neutral_scores:bool = False):
        """
        Compute document-level and topic-level features based on Zhang et al. 2018 --> classification into negative, neutral, positive; positive count / (positive + negative count) - we have now PageRank weighting
        filter_neutral_scores: boolean to decide if we want to consider neutral scores --> most scores are neutral. Thus, the scores are dragged towards 1, weakening the signal
        """
        #1.-------------Overall mean sentiment
        #1. get grouped df by document and per document-topic
        grouped, grouped_topic = self._group_df_(filter_neutral_scores = filter_neutral_scores, compute_shares = False)
        
        #2. Calculate mean sentiment based on definition in Zhang 2018
        grouped["mean_sentiment"] = grouped["pos"] / (grouped["pos"] + grouped["neg"] + eps)  

        #3. keep only relevant columns
        grouped = grouped[["doc_id", "mean_sentiment"]] #keep only doc_id (identifier) and mean_sentiment (value)

        #2.-----------------------Topic sentiment
        #1. Calculate mean sentiment following the definition of Zhang 2018
        grouped_topic["mean_sentiment"] = grouped_topic["pos"] / (grouped_topic["pos"] + grouped_topic["neg"] + eps) 

        #2. Pivot into wide format and rename columns
        grouped_topic = self._pivot_topic_(grouped_topic)

        #4.-----join mean sentiment and topic-sentiment in one df
        sentiment_scores = grouped.merge(grouped_topic, on = "doc_id")
        self.features = sentiment_scores
        return self

    def compute_aggregates_gupta_a(self, eps = 1e-5, filter_neutral_scores:bool = False):
        """
        Compute document-level and topic-level features based on Gupta et al. 2020a --> classification into negative, neutral, positive; positive count - negative count
        filter_neutral_scores: boolean to decide if we want to consider neutral scores --> most scores are neutral. Thus, the scores are dragged towards 1, weakening the signal
        """
        #1.-------------Overall mean sentiment
        #1. get grouped df by document and per document-topic
        grouped, grouped_topic = self._group_df_(filter_neutral_scores = filter_neutral_scores, compute_shares = False)
        
        #2. Calculate mean sentiment based on definition in Gupta et al. 2020a (definition 1)
        grouped["mean_sentiment"] = grouped["pos"] - grouped["neg"]  

        #3. keep only relevant columns
        grouped = grouped[["doc_id", "mean_sentiment"]] #keep only doc_id (identifier) and mean_sentiment (value)

        #2.-----------------------Topic sentiment
        #1. Calculate mean sentiment following the definition of Zhang 2018
        grouped_topic["mean_sentiment"] = grouped_topic["pos"] - grouped_topic["neg"]

        #2. Pivot into wide format and rename columns
        grouped_topic = self._pivot_topic_(grouped_topic)

        #4.-----join mean sentiment and topic-sentiment in one df
        sentiment_scores = grouped.merge(grouped_topic, on = "doc_id")
        self.features = sentiment_scores
        return self

    def compute_aggregates_gupta_b(self, eps = 1e-5, filter_neutral_scores:bool = False):
        """
        Compute document-level and topic-level features based on Gupta et al. 2020a --> classification into negative, neutral, positive; 
        positive count / all counts
        filter_neutral_scores: boolean to decide if we want to consider neutral scores --> most scores are neutral. Thus, the scores are dragged towards 1, weakening the signal
        """
        #1.-------------Overall mean sentiment
        #1. get grouped df by document and per document-topic
        grouped, grouped_topic = self._group_df_(filter_neutral_scores = filter_neutral_scores, compute_shares = False)
        
        #2. Calculate mean sentiment based on definition in Gupta et al. 2020a (definition 2)
        grouped["mean_sentiment"] = grouped["pos"] / (grouped["num_sentences"] + eps)

        #3. keep only relevant columns
        grouped = grouped[["doc_id", "mean_sentiment"]] #keep only doc_id (identifier) and mean_sentiment (value)

        #2.-----------------------Topic sentiment
        #1. Calculate mean sentiment following the definition of Zhang 2018
        grouped_topic["mean_sentiment"] = grouped_topic["pos"] / (grouped_topic["num_sentences"] + eps)

        #2. Pivot into wide format and rename columns
        grouped_topic = self._pivot_topic_(grouped_topic)
        
        #3.-----join mean sentiment and topic-sentiment in one df
        sentiment_scores = grouped.merge(grouped_topic, on = "doc_id")
        self.features = sentiment_scores
        return self

    def compute_aggregates_gupta_c(self, eps = 1e-5, filter_neutral_scores:bool = False):
        """
        Compute document-level and topic-level features based on Gupta et al. 2020a --> classification into negative, neutral, positive; 
        (positive count - negative counts) / all counts
        filter_neutral_scores: boolean to decide if we want to consider neutral scores --> most scores are neutral. Thus, the scores are dragged towards 1, weakening the signal
        """
        #1.-------------Overall mean sentiment
        #1. get grouped df by document and per document-topic
        grouped, grouped_topic = self._group_df_(filter_neutral_scores = filter_neutral_scores, compute_shares = False)
        
        #2. Calculate mean sentiment based on definition in Gupta et al. 2020a (definition 2)
        grouped["mean_sentiment"] = (grouped["pos"] - grouped["neg"]) / (grouped["num_sentences"] + eps)

        #3. keep only relevant columns
        grouped = grouped[["doc_id", "mean_sentiment"]] #keep only doc_id (identifier) and mean_sentiment (value)

        #2.-----------------------Topic sentiment
        #1. Calculate mean sentiment following the definition of Zhang 2018
        grouped_topic["mean_sentiment"] = (grouped_topic["pos"] - grouped_topic["neg"]) / (grouped_topic["num_sentences"] + eps)

        #2. Pivot into wide format and rename columns
        grouped_topic = self._pivot_topic_(grouped_topic)

        #3.-----join mean sentiment and topic-sentiment in one df
        sentiment_scores = grouped.merge(grouped_topic, on = "doc_id")
        self.features = sentiment_scores
        return self

    def compute_aggregates_renault(self, filter_neutral_scores: bool = True):
        """
        Compute document-level and topic-level features based on Renault 2017 --> classification into negative, neutral, positive; 
        Problem: Different contributions of documents/topics of differing sizes
        Sum over (sentiment_class contribution × sentiment_prob).
        
        - Positive sentences contribute +probability
        - Negative sentences contribute -probability
        - Neutral sentences contribute 0 (can optionally be filtered out)
        """
        if self.df is None:
            raise ValueError("DataFrame not built. Call build_dataframe() first.")

        df = self.df.copy()

        #1. Assign contribution score: -p for neg, +p for pos, 0 for neutral
        df["sentiment_contrib"] = 0.0
        df.loc[df["sentiment_class"] == 0, "sentiment_contrib"] = -df["sentiment_prob"] #--> -p for negatives
        df.loc[df["sentiment_class"] == 2, "sentiment_contrib"] = df["sentiment_prob"] #--> +p for positives

        #decide if we filter neutral scores
        if filter_neutral_scores:
            df = df[df["sentiment_class"] != 1].reset_index(drop=True)

        #2.---- Document-level aggregate
        grouped = df.groupby("doc_id")["sentiment_contrib"].mean().reset_index() #compute mean sentiment contribution per document
        grouped = grouped.rename(columns={"sentiment_contrib": "mean_sentiment"}) #rename column to mean_sentiment

        #3.---- Topic-level aggregate
        grouped_topic = (
            df.groupby(["doc_id", "topic"])["sentiment_contrib"].mean().reset_index() #compute mean sentiment contribution per document and topic
        )

        #pivot into wide format
        grouped_topic = self._pivot_topic_(grouped_topic, value_col="sentiment_contrib")

        # ---- Merge overall + per-topic
        sentiment_scores = grouped.merge(grouped_topic, on="doc_id")
        self.features = sentiment_scores
        return self

    def compute_aggregates_vamossy(self, eps = 1e-5, filter_neutral_scores: bool = True):
        """
        Compute document-level and topic-level features based on Vamossy 2021.
        
        Definition:
        Sentiment = (positive messages * prob) - (negative messages * prob)
        Neutral = 0
        Scale by (1 + sum of probabilities) to normalize confidence mass.
        """
        if self.df is None:
            raise ValueError("DataFrame not built. Call build_dataframe() first.")
    
        df = self.df.copy()
    
        #-------1. Assign contribution score: -p for neg, +p for pos, 0 for neutral
        df["sentiment_contrib"] = 0.0
        df.loc[df["sentiment_class"] == 0, "sentiment_contrib"] = -df["sentiment_prob"] #--> -p for negatives
        df.loc[df["sentiment_class"] == 2, "sentiment_contrib"] = df["sentiment_prob"] #--> +p for positives

        #decide if we filter neutral scores
        if filter_neutral_scores:
            df = df[df["sentiment_class"] != 1].reset_index(drop=True)
    
        #2.---------- Document-level aggregate ----------
        doc_group = df.groupby("doc_id").agg(
            contrib_sum=("sentiment_contrib", "sum"),
            prob_sum=("sentiment_prob", "sum"),
            n=("sentiment_contrib", "count")
        ).reset_index()
    
        #scale by confidence
        doc_group["mean_sentiment"] = doc_group["contrib_sum"] / (1 + doc_group["prob_sum"] + eps)

        #keep only relevant columns
        doc_group = doc_group[["doc_id", "mean_sentiment"]]
    
        #3.---------- Topic-level aggregate ----------
        topic_group = df.groupby(["doc_id", "topic"]).agg(
            contrib_sum=("sentiment_contrib", "sum"),
            prob_sum=("sentiment_prob", "sum"),
            n=("sentiment_contrib", "count")
        ).reset_index()
    
        topic_group["mean_sentiment"] = topic_group["contrib_sum"] / (1 + topic_group["prob_sum"] + eps)
        
        #pivot and rename using helper
        topic_group = self._pivot_topic_(topic_group, value_col="mean_sentiment")
    
        # ---------- Merge ----------
        sentiment_scores = doc_group.merge(topic_group, on="doc_id")
        self.features = sentiment_scores
        return self


    def _merge_meta_(self, drop_cols=None, drop_dupes_subset=None):
        """
        helper method to merge self.features with metadata and ticker mapping.
        drop_cols: list of columns to drop after merge
        drop_dupes_subset: list of columns to drop duplicates on
        drop rows where year_of_report is missing (only 1 case/accession number)
        """
        if self.features is None:
            raise ValueError("Features not computed. Run compute_* before merging metadata.")

        #create df from metadata
        meta_df = pd.DataFrame(self.metadata)

        #map cik → ticker (vectorized)
        meta_df["ticker"] = meta_df["cik"].map(self.cik_ticker_mapping)

        #clean & format columns
        meta_df = meta_df[["doc_id", "ticker", "filing_date", "cik", "year_of_report"]] #extract only relevant columns
        meta_df["year_of_report"] = pd.to_numeric(meta_df["year_of_report"], errors="coerce")
        meta_df = meta_df.dropna(subset=["year_of_report"])  #handle missing years
        meta_df["year_of_report"] = meta_df["year_of_report"].astype(int) #convert to int safely (after dropping NaNs)
        meta_df["filing_date"] = pd.to_datetime(meta_df["filing_date"], format="%Y%m%d", errors = "coerce") #transform to date format

        #drop duplicate doc_ids
        meta_df = meta_df.drop_duplicates(subset="doc_id").reset_index(drop=True)

        #merge features with metadata
        text_features = self.features.merge(meta_df, on="doc_id")

        #drop duplicates if requested
        if drop_dupes_subset is not None:
            text_features = text_features.drop_duplicates(subset=drop_dupes_subset).reset_index(drop=True)

        #drop columns if requested
        if drop_cols is not None:
            text_features = text_features.drop(columns=drop_cols)

        self.features = text_features
        return self.features

    def add_ticker_year_stock(self):
        """
        Add ticker and year for stock prediction tasks.
        Drops cik column, keeps filing_date.
        Ensures unique (ticker, year_of_report).
        """
        return self._merge_meta_(
            drop_cols=["cik"], 
            drop_dupes_subset=["ticker", "year_of_report"]
        )
        
    def add_ticker_year_fin(self):
        """
        Add ticker and year for financial prediction tasks.
        Drops cik and filing_date, keeps only year_of_report.
        Ensures unique (ticker, year_of_report).
        """
        return self._merge_meta_(
            drop_cols=["cik", "filing_date"], 
            drop_dupes_subset=["ticker", "year_of_report"]
        )

 
    def run_stock(self, filter_neutral_scores:bool = True):
        """
        Combined method to create the text features for stock prediction tasks
        """
        self.build_dataframe()
        self.compute_aggregates(filter_neutral_scores = filter_neutral_scores)
        self.get_features()
        
        return self.add_ticker_year_stock()

    def run_fin(self, filter_neutral_scores:bool = True):
        """
        Combined method to create the text features for financial prediction tasks
        """
        self.build_dataframe()
        self.compute_aggregates(filter_neutral_scores = filter_neutral_scores)
        self.get_features()
        
        return self.add_ticker_year_fin()

    def run_stock_azimi(self, filter_neutral_scores:bool = False):
        """
        Combined method to create the text features for stock prediction tasks - based on Azimi et al. 2021 --> share of positive / share of negative sentences
        """
        self.build_dataframe()
        self.compute_aggregates_azimi(filter_neutral_scores = filter_neutral_scores)
        self.get_features()
        
        return self.add_ticker_year_stock()

    def run_fin_azimi(self, filter_neutral_scores:bool = False):
        """
        Combined method to create the text features for financial prediction tasks - based on Azimi et al. 2021 --> share of positive / share of negative sentences
        """
        self.build_dataframe()
        self.compute_aggregates_azimi(filter_neutral_scores = filter_neutral_scores)
        self.get_features()
        
        return self.add_ticker_year_fin()

    def run_all_stock(self, filter_neutral_scores=False):
        """
        Run all aggregation methods for stock prediction, add ticker/year metadata,
        and keep each aggregation as a separate DataFrame in a dict.
        """
        self.build_dataframe()
        
        methods = {
            "basic": self.compute_aggregates,
            "azimi": self.compute_aggregates_azimi,
            "huang": self.compute_aggregates_huang,
            "zhang": self.compute_aggregates_zhang,
            "gupta_a": self.compute_aggregates_gupta_a,
            "gupta_b": self.compute_aggregates_gupta_b,
            "gupta_c": self.compute_aggregates_gupta_c,
            "renault": self.compute_aggregates_renault,
            "vamossy": self.compute_aggregates_vamossy
        }
        
        results = {}
        #iterate over each aggregation technique and the corresponding method
        for name, method in tqdm(methods.items(), desc = "Progress Stock"):
            print(f"Aggregation method: {name}")
            method(filter_neutral_scores=filter_neutral_scores)  # run aggregation
            features = self.get_features()                        # get the features
            features = self.add_ticker_year_stock()               # add ticker/year for stock
            results[name] = features
            print(f"Shape of df: {features.shape}\n")
        
        return results


    def run_all_fin(self, filter_neutral_scores=False):
        """
        Run all aggregation methods for financial prediction, add ticker/year metadata,
        and keep each aggregation as a separate DataFrame in a dict.
        """
        self.build_dataframe()
        
        methods = {
            "basic": self.compute_aggregates,
            "azimi": self.compute_aggregates_azimi,
            "huang": self.compute_aggregates_huang,
            "zhang": self.compute_aggregates_zhang,
            "gupta_a": self.compute_aggregates_gupta_a,
            "gupta_b": self.compute_aggregates_gupta_b,
            "gupta_c": self.compute_aggregates_gupta_c,
            "renault": self.compute_aggregates_renault,
            "vamossy": self.compute_aggregates_vamossy
        }
        
        results = {}
        #iterate over each aggregation technique and the corresponding method
        for name, method in tqdm(methods.items(), desc = "Progress Fin."):
            print(f"Aggregation method: {name}")
            method(filter_neutral_scores=filter_neutral_scores)  # run aggregation
            features = self.get_features()                        # get the features
            features = self.add_ticker_year_fin()                 # add ticker/year for financial
            results[name] = features
            print(f"Shape of df: {features.shape}\n")
        
        return results