import pandas as pd
import numpy as np
import pickle
import sys
sys.path.insert(0, "/mnt/sdb1/home/simonj/Paper 2")
import utils_rag


#define class for loading and preprocessing the data
class DataPreprocessor:
    def __init__(self):
        """
        Initialization
        """

    def load_metadata(self, metadata_path, text_name:str = "texts", metadata_name:str = "metadata"):
        """
        Load Item 7 metadata and texts.
        """
        with open(metadata_path, "rb") as file:
            data = pickle.load(file)
            
        texts = data[text_name]
        metadata = data[metadata_name]
        
        print("Metadata and texts loaded.\n")
        return texts, metadata
        


    def load_knn_indices(self, knn_path, model_name:str = "acl_modern_bert"):
        """
        Load KNN-assigned pseudo topic labels.
        """
        with open(knn_path, "rb") as file:
            knn = pickle.load(file)

        print("KNN data loaded.\n")
        return knn[model_name]["indices"]

    def load_rag_results(self, rag_path):
        """
        Load the rag-llm transformed topics.
        """
        with open(rag_path, "rb") as file:
            rag = pickle.load(file)
        print("RAG data loaded.\n")
        return rag["indices_to_check"], rag["answers"]

    def load_modalities(self, modalities_path):
        """
        Load the sentiment scores.
        """
        with open(modalities_path, "rb") as file:
            data = pickle.load(file)
        
        sentiment_preds = data["sentiment"]
        fls_results = data["fls"]
        print("Sentiment and FLS data loaded.\n")
        return sentiment_preds, fls_results


    def load_embeddings(self, embedding_path, model_name):
        """
        Load embedding matrix
        """
        with open(embedding_path, "rb") as file:
            data = pickle.load(file)
            
        embeddings = data[model_name]
        print(f"Embeddings loaded. Shape: {embeddings.shape}")
        return embeddings

    def load_llm_sentiment(self, llm_sentiment_path):
        "Load LLM sentiment"
        with open(llm_sentiment_path, "rb") as file:
            data = pickle.load(file)

        sentiment_results = data["results"]
        sentiment_df = data["df"]
        print("LLM sentiment loaded")
        return sentiment_df, sentiment_results
    
    def build_grouped_df(self, item7_metadata_merged, item7_texts, indices, combined_topics):
        """
        Build a grouped dataframe based on metadata, topics, etc.
        Returns:
            dataframe with text grouped by document and assigned topic
        """
        df = pd.DataFrame(item7_metadata_merged)
        df["sentence"] = item7_texts
        df["topic_int"] = indices

        combined_topics_dict = {-1: "noise"}
        for i, topic in enumerate(combined_topics):
            combined_topics_dict[i] = topic

        df["topic_name"] = df["topic_int"].map(combined_topics_dict)
        grouped = df.groupby(["doc_id", "topic_int", "company_name", "year_of_report", "topic_name"])["sentence"].apply(lambda s: "; ".join(s)).reset_index()
        print("Grouped df created. \n")
        return grouped