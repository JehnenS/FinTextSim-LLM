from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm

rel_cols = ['close_movement_class', 'total_return', 'total_volume', 'avg_volume',
       'volatility_volume', 'volatility_adjClose', 'max_drawdown', 'SMA_10',
       'SMA_50', 'EMA_10', 'EMA_50', 'BB_Distance_Upper', 'RSI', 'MACD_Hist',
       'Volume_SMA_50', 'Volume_SMA_10']



from collections import defaultdict
import torch

def init_doc_entry():
    """
    Returns the initial structure for each document entry.
    """
    return {
        "topics": [],
        "sentiments": [],
        "fls_labels": [],
    }

def prepare_sentence_level_input(topics, sentiments, fls_labels, meta_merged):
    """
    Groups sentence-level data by document ID

    Args:
        topics (List[int]): Topic labels per sentence.
        sentiments (List[int]): Sentiment labels per sentence.
        fls_labels (List[int]): FLS labels per sentence.
        meta_merged (List[dict]): Metadata per sentence.

    Returns:
        dict: doc_id → {
            'topics': [int, ...],
            'sentiments': [int, ...],
            'fls_labels': [int, ...],
        }
    """
    doc_data = defaultdict(init_doc_entry)

    for topic, sent, fls, meta in zip(topics, sentiments, fls_labels, meta_merged):
        doc_id = meta["doc_id"] #extract doc_id
        topic = 14 if topic == -1 else topic #assign 14 to noise topic as one-hot encoding does not allow negative values

        
        doc_data[doc_id]["topics"].append(int(topic))
        doc_data[doc_id]["sentiments"].append(int(sent[0])) #append sentiment label only
        doc_data[doc_id]["fls_labels"].append(int(fls[0])) #append fls label only

    print("Sentence input data prepared.")

    return dict(doc_data)  # Convert defaultdict to dict for pickling


def init_topic_entry():
    """
    Returns the initial structure for each document entry.
    """
    return {
        "topic_ids": [],
        "sentiments": [],
        "coherences": [],
    }


def prepare_topic_level_input(topic_df):
    """
    Groups topic-level data by document ID.

    Args:
        topic_df (pd.DataFrame): Must contain 'doc_id', 'topic_int', 'sentiment', and 'coherence' columns.

    Returns:
        dict: doc_id → {
            'topic_ids': [int, ...],
            'sentiments': [int, ...],
            'coherences': [int, ...],
        }
    """
    doc_data = defaultdict(init_topic_entry)

    

    for _, row in topic_df.iterrows():
        doc_id = row["doc_id"]
        topic_id = row["topic_int"]
        topic_id = 14 if topic_id == -1 else topic_id
        sentiment = row["sentiment"]
        coherence = row["coherence"]

        doc_data[doc_id]["topic_ids"].append(int(topic_id))
        doc_data[doc_id]["sentiments"].append(int(sentiment))
        doc_data[doc_id]["coherences"].append(int(coherence))

    print("Topic input data prepared.")

    return dict(doc_data)







def prepare_document_level_input(fin_df, rel_cols:list):
    """
    Groups document-level financials

    Args:
        fin_df: pd.DF: df with financial predictors (and target) per document
        rel_cols (list): Financial columns to include, including the target.

    Returns:
        dict: doc_id → {
            'topics': [int, ...],
            'sentiments': [int, ...],
            'fls_labels': [int, ...],
        }
    """
    doc_data = defaultdict(dict)

    for _, row in fin_df.iterrows():
        doc_id = row["doc_id"] #extract doc_id
        for col in rel_cols: #iterate over the relevant columns
            metric_value = row[col]  # Dynamic attribute access
            doc_data[doc_id][col] = metric_value

    print("Document input data prepared")
    
    return dict(doc_data)  # Convert defaultdict to dict for pickling