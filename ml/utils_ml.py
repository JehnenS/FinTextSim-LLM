import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm

categorical_cols = [
    #"cap_category_label"
]
cols_to_exclude = ["fiscal_year", "fiscal_year_growth", "year_x", "year_y", "fiscal_year_x", "fiscal_year_y", "fiscal_year_growth_x", "fiscal_year_growth_y", "doc_id", "year_of_report", "index", "pred_year", "key_date"]
models = ["mistral:7b-instruct", "llama3.2:3b-instruct-q8_0", "deepseek-r1:7b"] #define the models/runs across we want to ensemble the LLM results

text_only_cols = ['mean_sentiment', 'mean_sentiment_topic_10', 'mean_sentiment_topic_0', 'mean_sentiment_topic_3', 'mean_sentiment_topic_11', 'mean_sentiment_topic_5', 'mean_sentiment_topic_1', 'mean_sentiment_topic_2', 'mean_sentiment_topic_9', 'mean_sentiment_topic_4', 'mean_sentiment_topic_6', 'mean_sentiment_topic_8', 'mean_sentiment_topic_7'] #, 'mean_sentiment_topic_-1'


def load_result_loc(result_dict, model_name, base_path = "paper2/Results/ML/Financials"):
    """
    Load results loc based on configuration of the ML models
    """
    config = result_dict["config"]
    # Determine prefixes and suffixes
    prefix = f"{model_name}_swade" if config.get("feature_set") == "swade" else f"{model_name}"
    suffix = "_cik_symbol" if config.get("sample") == "cik_symbol" else ""
    suffix2 = "" if config.get("exclude_quarter_features") else "_incl_q_feats"
    
    # Compose filenam
    result_loc = (
        f"{base_path}/"
        f"{prefix}_{config.get('target_period')}_{config.get('fintextsim')}{suffix}{suffix2}.pkl"
    )

    print(f"Results will be saved to {result_loc}")
    return result_loc

def load_result_loc_stock(result_dict, model_name, base_path = "paper2/Results/ML/Financials"):
    """
    Load results loc based on configuration of the ML models
    """
    config = result_dict["config"]
    # Determine prefixes and suffixes
    prefix = f"{model_name}_swade" if config.get("feature_set") == "swade" else f"{model_name}"
    suffix = "_cik_symbol" if config.get("sample") == "cik_symbol" else ""
    
    # Compose filename
    result_loc = (
        f"{base_path}/"
        f"{prefix}_{config.get('fintextsim')}{suffix}.pkl"
    )

    print(f"Results will be saved to {result_loc}")
    return result_loc


def load_data_loc(result_dict, base_path = "paper2/Data/FMP/fmp_data"):
    """
    Load data loc based on configuration of the ML models
    """
    config = result_dict["config"]
    # Determine prefixes and suffixes
    suffix = "_cik_symbol" if config.get("sample") == "cik_symbol" else ""
    
    # Compose filename
    data_loc = (f"{base_path}{suffix}.pkl")

    print(f"Load FMP data from {data_loc}")
    return data_loc

def load_sentiment_loc(result_dict, base_path = "paper2/Data/Features/text_features/sentiment_transformers"):
    """
    Load sentiment loc based on configuration of the ML models
    """
    config = result_dict["config"]
    # Determine prefixes and suffixes
    suffix = "_cik_symbol" if config.get("sample") == "cik_symbol" else "_fd_sp500"
    suffix2 = "_quantile" if config.get("outlier_detection") == "quantile" else ""
    
    # Compose filename
    data_loc = (f"{base_path}{suffix}{suffix2}.pkl")

    print(f"Load Sentiment data from {data_loc}")
    return data_loc


def one_hot_encode(index, num_classes):
    """
    One hot encoding - particularly for sentiment, topics and FLS
    """
    return F.one_hot(torch.tensor(index), num_classes=num_classes).float()



import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    Collate function to batch variable-length documents - e.g. documents have differing number of sentences, embeddings, topic assignment, etc.

    Args:
        batch (list of dict): Each dict is output of __getitem__ from FinancialDocDataset.

    Returns:
        dict: Batched and padded tensors.
    """

    # Extract fields from batch
    doc_ids = [item["doc_id"] for item in batch]

    embeddings_list = [item["embeddings"] for item in batch]  # List of [num_sentences, emb_dim]
    topics_list = [item["topics"] for item in batch]          # List of [num_sentences, topic_dim] or [num_sentences]
    sentiments_list = [item["sentiments"] for item in batch]
    fls_list = [item["fls_labels"] for item in batch]

    financial_vecs = torch.stack([item["financial_vec"] for item in batch])  # [batch_size, num_fin_features]
    labels = torch.stack([item["label"] for item in batch])                 # [batch_size]

    # Lengths of each document (number of sentences)
    lengths = torch.tensor([emb.shape[0] for emb in embeddings_list])

    # Pad embeddings (pad sequences to max length in batch along sentence dimension)
    embeddings_padded = pad_sequence(embeddings_list, batch_first=True)  # [batch_size, max_seq_len, emb_dim]

    # Pad sentence-level labels (topics, sentiments, fls)
    # If one-hot encoded, shape is [num_sentences, feature_dim], else [num_sentences]
    # pad_sequence works on [seq_len, *] so transpose is necessary if feature_dim > 1

    def pad_labels(label_list):
        # Check if labels are 1D or 2D (one-hot)
        if label_list[0].dim() == 1:
            # 1D: e.g. topics as indices: pad with a value (e.g. -1) for missing
            return pad_sequence(label_list, batch_first=True, padding_value=-1)
        else:
            # 2D one-hot: shape = [seq_len, feature_dim]
            # We need to pad along seq_len dimension:
            max_len = max(l.shape[0] for l in label_list)
            feature_dim = label_list[0].shape[1]
            padded = torch.zeros((len(label_list), max_len, feature_dim))
            for i, l in enumerate(label_list):
                padded[i, :l.shape[0], :] = l
            return padded

    topics_padded = pad_labels(topics_list)
    sentiments_padded = pad_labels(sentiments_list)
    fls_padded = pad_labels(fls_list)

    return {
        "doc_id": doc_ids,
        "embeddings": embeddings_padded,
        "topics": topics_padded,
        "sentiments": sentiments_padded,
        "fls_labels": fls_padded,
        "financial_vec": financial_vecs,
        "label": labels,
        "lengths": lengths
    }

from sklearn.model_selection import train_test_split

def split_doc_data(doc_data, label_name="close_movement_class", test_size=0.2, random_state=42):
    """
    Split the data into test- and trainset --> make sure that we use stratified splitting to ensure balanced learning
    """
    doc_ids = list(doc_data["document_data"].keys())
    labels = [doc_data["document_data"][doc_id][label_name] for doc_id in doc_ids]

    # Stratified split
    train_ids, test_ids = train_test_split(
        doc_ids, 
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    # Build the subsets
    train_data_sentence = {doc_id: doc_data["sentence_data"][doc_id] for doc_id in train_ids}
    train_data_topics = {doc_id: doc_data["topic_data"][doc_id] for doc_id in train_ids}
    train_data_documents = {doc_id: doc_data["document_data"][doc_id] for doc_id in train_ids}

    train_data = {
        "sentence_data": train_data_sentence,
        "topic_data": train_data_topics,
        "document_data": train_data_documents
    }
    
    test_data_sentence = {doc_id: doc_data["sentence_data"][doc_id] for doc_id in test_ids}
    test_data_topics = {doc_id: doc_data["topic_data"][doc_id] for doc_id in test_ids}
    test_data_documents = {doc_id: doc_data["document_data"][doc_id] for doc_id in test_ids}

    test_data = {
        "sentence_data": test_data_sentence,
        "topic_data": test_data_topics,
        "document_data": test_data_documents
    }
    return train_data, test_data

def collate_fn(batch):
    """
    New collate fn
    """
    from torch.nn.utils.rnn import pad_sequence

    sentence_inputs = [item['sentence_input'] for item in batch]
    topic_inputs = [item['topic_input'] for item in batch]
    financials = torch.stack([item['financials'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])

    sentence_padded = pad_sequence(sentence_inputs, batch_first=True, padding_value=0.0)
    topic_padded = pad_sequence(topic_inputs, batch_first=True, padding_value=0.0)

    return {
        'sentence_input': sentence_padded,
        'topic_input': topic_padded,
        'financials': financials,
        'target': targets
    }