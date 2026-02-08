import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from collections import defaultdict
import re


class TextFeatureCreator:
    def __init__(self, 
                 texts, 
                 metadata, 
                 topics,
                sentiments,
                fls):
        self.texts = texts
        self.metadata = metadata
        self.topics = topics

        self.sentiments = sentiments
        self.fls = fls


    def _get_text_stats_(self):
        """
        Function to obtain basic text statistics aggregated across document-id --> word counts, number of sentences, ratio of discussed topics
        """
            
        doc_stats = defaultdict(lambda: {
            "num_sentences": 0,
            "num_words": 0,
            "word_counts": [],
            "topic_counts": defaultdict(int)
        })

        #zip texts, meta and topics and iterate over them
        for text, meta, topic in tqdm(zip(self.texts, self.metadata, self.topics), desc = "Compute Scores", total = len(self.texts)):
            doc_id = meta.get("doc_id")
            if doc_id is None:
                continue  # skip if no doc_id
                    
            word_count = len(text.split()) #get word count by splitting at whitespaces
        
            doc_stats[doc_id]["num_sentences"] += 1 #add 1 to sentence counter
            doc_stats[doc_id]["num_words"] += word_count #add number of words
            doc_stats[doc_id]["word_counts"].append(word_count) #add word count - e.g.measure how different sentence lengths are - average and/or std
            doc_stats[doc_id]["topic_counts"][topic] += 1 #add the count for the topic
        
        #convert doc_stats to DataFrame - evaluation per doc-id
        records = []
        for doc_id, stats in tqdm(doc_stats.items(), desc = "Transform to df"):
            num_sentences = stats["num_sentences"]
            num_words = stats["num_words"]
            word_counts = stats["word_counts"]
        
            avg_sentence_length = num_words / num_sentences if num_sentences else 0
            std_sentence_length = np.std(word_counts) if word_counts else 0
        
            record = {
                "doc_id": doc_id,
                "num_sentences": num_sentences,
                "num_words": num_words,
                "avg_sentence_length": avg_sentence_length,
                "std_sentence_length": std_sentence_length
            }
        
            total = sum(stats["topic_counts"].values())
            for topic_id, count in stats["topic_counts"].items():
                record[f"topic_ratio_{topic_id}"] = count / total if total > 0 else 0
        
            records.append(record)
        
        return pd.DataFrame(records)

    def _aggregate_doc_features_(self):
        """
        Function to obtain aggregate scores based on document id - includes sentiment, FLS, topics (standalone) and the combination
        """
        rows = []
        doc_ids = [entry.get("doc_id") for entry in self.metadata]
        
        for (sent, fls, topic, doc_id, text) in tqdm(zip(self.sentiments, self.fls, self.topics, doc_ids, self.texts), desc="Create df"):
            class_id, sent_prob = sent
            fls_label, fls_prob = fls # extract individuals from fls
    
    
            rows.append({
                "doc_id": doc_id,
                "sentiment_class": class_id,
                "sentiment_prob": sent_prob,
                "fls_label": fls_label,
                "fls_prob": fls_prob,
                "topic": topic,
                
            })
    
        df = pd.DataFrame(rows)
        results = []

        #calculate scores grouped by doc-id
        for doc_id, group in tqdm(df.groupby("doc_id"), desc="Compute scores"):
            result = {"doc_id": doc_id}
    
            # Global Sentiment
            sent_counts = group["sentiment_class"].value_counts(normalize=True)
            for cls in [0, 1, 2]:
                result[f"sentiment_ratio_{cls}"] = sent_counts.get(cls, 0)
            result["avg_sentiment_prob"] = group["sentiment_prob"].mean()
    
            # Global FLS
            fls_counts = group["fls_label"].value_counts(normalize=True)
            for label in [0, 1, 2]:
                 result[f"fls_ratio_{key}"] = fls_counts.get(label, 0)
            result["avg_fls_prob"] = group["fls_prob"].mean()
    
            # Sentiment × FLS
            for sent_cls in [0, 1, 2]:
                for fls_label in ["Specific FLS", "Non-specific FLS"]:
                    cond = (group["sentiment_class"] == sent_cls) & (group["fls_label"] == fls_label)
                    result[f"sentiment_{sent_cls}_fls_{fls_label.replace(' ', '_').lower()}_ratio"] = cond.mean()
    
            # Topic-Level Aggregates
            for topic, topic_group in group.groupby("topic"):
                topic_sent_counts = topic_group["sentiment_class"].value_counts(normalize=True)
                for cls in [0, 1, 2]:
                    result[f"sentiment_ratio_{cls}_topic_{topic}"] = topic_sent_counts.get(cls, 0)
    
                topic_fls_counts = topic_group["fls_label"].value_counts(normalize=True)
                for label in [0, 1, 2]:
                    result[f"fls_ratio_{label}_topic_{topic}"] = topic_fls_counts.get(label, 0)
    
                for sent_cls in [0, 1, 2]:
                    for fls_label in ["Specific FLS", "Non-specific FLS"]:
                        cond = (topic_group["sentiment_class"] == sent_cls) & (topic_group["fls_label"] == fls_label)
                        result[f"sentiment_{sent_cls}_fls_{fls_label.replace(' ', '_').lower()}_ratio_topic_{topic}"] = cond.mean()
    
            results.append(result)
    
        #get other text-based features
    
        return pd.DataFrame(results)