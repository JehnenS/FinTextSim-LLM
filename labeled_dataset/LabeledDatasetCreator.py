import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from collections import Counter

class LabeledDatasetCreator:
    """
    Class to create a labeled dataset from sentences based on keyword list
    Distinction to Paper 1 and Item7: no metadata
    """

    def __init__(self, 
                 sentences: list,
                 topic_names:list,
                 keyword_list:list, 
                 keyword_blacklist:list,
                 exclusion_dict:dict = None):
        
        self.sentences = [sent.lower() for sent in sentences]
        self.topic_names = topic_names
        self.keyword_list = [[word.lower() for word in sublist] for sublist in keyword_list]
        self.keyword_blacklist = set(word.lower() for word in keyword_blacklist) #ensure lowercase and no duplicates
        self.exclusion_dict = exclusion_dict

        #flatten the keyword list from list of keywords per topic to flat list
        self.keyword_list_flat = [word for sublist in self.keyword_list for word in sublist]

        print(f"Number of keywords: {len(self.keyword_list_flat)}")
        print(f"Number of topics: {len(self.keyword_list)}")
    
        #check blacklist of keywords
        print(f"Number of blacklisted words (substring): {len(self.keyword_blacklist)}")



    def create_keyword_sent_matrix(self, exact_match = False):
        """
        Function to use as base for creating topic labels 
        Occurrence-matrix of keywords in sentences
    
        input:
        sentences: List of sentences - words
        keywords: List of keywords
        keyword_blacklist: list of keywords which should not be considered --> e.g. "taxonomy" contains "tax" but is not relevant for the tax and regulations topic
        exact_match: bool if the keyword has to appear exactly or as part of a word --> example: "operation": part of "operations" --> if exact match, it would not be recognized
        """

        
        # Initialize a matrix to store the occurrences of keywords in each sentence
        matrix = np.zeros((len(self.sentences), len(self.keyword_list_flat)), dtype=int)
    
    
        #iterate over each topic
        for i, sentence in enumerate(tqdm(self.sentences, desc = "Keyword-Sentence-Matrix progress")):
            #Split sentence into words
            sentence_words = sentence.lower().split()
            
            #iterate over each keyword
            for j, keyword in enumerate(self.keyword_list_flat):
                if exact_match:
                    count = sum(1 for word in sentence_words if word == keyword)
                else:    
                    # Count the occurrences of the keyword in the topic, considering substrings
                    count = sum(1 for word in sentence_words if (keyword in word) and not any(blacklisted in word for blacklisted in self.keyword_blacklist))
                
                # Append the count to the keyword_counts list
                matrix[i, j] = count
    
        print(f"Keyword-Sent Matrix created.\n Shape: {matrix.shape}")
        return matrix
    
    def analyze_keyword_distribution(self, keyword_sent_matrix):
        """
        Analyze the distribution of keywords per topic --> check to which topic the matched keyword belongs --> column in the keyword-sentence-matrix
    
        Input:
        - keyword_sent_matrix: matrix of sentences and the keyword matches (output from create_keyword_sent_matrix) - dimensions: n_docs x n_keywords
        combined_keywords: list of lists containing the keywords per topic
        """
        # 1. Get the number of keywords per topic --> length of each inner list
        keywords_per_topic = [len(self.keyword_list[i]) for i in range(len(self.keyword_list))]
    
        # 2. Expand the keywords_per_topic list to match each column of keyword_topic_matrix
        topic_assignment = []
        for topic, num_keywords in enumerate(keywords_per_topic):
            topic_assignment.extend([topic] * num_keywords)
    
        # 3. Create lists to store results
        no_keywords = []
        one_topic = []
        one_topic_multiple_keywords = []
        multiple_topics = []
        full_keyword_sent_assignments = []  # This will store the topic assignments for each sentence
    
        # 4. Iterate over each row
        num_keyword_sent = keyword_sent_matrix.sum(axis=1)  # get the number of keywords per topic (rowsums)
        for i, num_keywords in enumerate(tqdm(num_keyword_sent, desc="Keyword Distribution Analysis")):
            if num_keywords > 0:
                # Get the topic assignments for the current row
                row_topic_assignments = [
                    topic_assignment[j] 
                    for j, count in enumerate(keyword_sent_matrix[i]) 
                    if count > 0
                ]
                # Ensure all matches are considered
                all_topic_assignments = []
                for j, count in enumerate(keyword_sent_matrix[i]):
                    all_topic_assignments.extend([topic_assignment[j]] * count)
    
                # Store the topic assignment of each keyword in this row
                full_keyword_sent_assignments.append((i, all_topic_assignments))
                
                # Check if only keywords from one topic are in the topic
                if len(set(all_topic_assignments)) == 1:
                    one_topic.append((i, all_topic_assignments))
                    if len(all_topic_assignments) > 1:
                        one_topic_multiple_keywords.append((i, all_topic_assignments))
                else:
                    multiple_topics.append((i, all_topic_assignments))
            else:
                no_keywords.append((i, None))
                full_keyword_sent_assignments.append((i, []))
    
            #print(f"Number of keywords in topic {i}: {num_keyword_topic[i]}")
    
        print(f"Number of topics containing no keyword: {len(no_keywords)}")
        print(f"Number of topics containing keywords from one topic: {len(one_topic)}")
        print(f"Number of topics containing multiple keywords from just one topic: {len(one_topic_multiple_keywords)}")
        print(f"Number of topics containing keywords from multiple topics: {len(multiple_topics)}")
        
        return full_keyword_sent_assignments
    
    
    def topic_assignment_keywords(self, topic_assignments, min_topic_words:int=2, max_words_other_topics:int=0):
        """
        Assigns a dominant topic to each sentence based on keyword assignments.
    
        Args:
        - topic_assignments: List of tuples where each tuple contains:
            - The index of the sentence.
            - A list of topics assigned to keywords in that sentences.
        - topic_names: List of topic names corresponding to the topic indices.
        - min_topic_words: Minimum number of keywords from a topic for it to be considered dominant.
        - max_words_other_topics: Maximum number of keywords from other topics allowed for the dominant topic.
    
        Returns:
        - A list of tuples where each tuple contains:
            - The topic representation index.
            - The assigned dominant topic (or None if no topic meets the criteria).
        """
        assigned_topics = []
    
        for topic_rep_index, keyword_assignments in topic_assignments:
            # Count occurrences of each topic in the keyword assignments
            topic_counts = Counter(keyword_assignments)
    
            # Get the total number of keywords
            total_keywords = len(keyword_assignments)
    
            # Check if there's a valid dominant topic based on the given conditions
            for topic, count in topic_counts.items():
                # Check if the dominant topic has at least `min_topic_words` keywords
                if count >= min_topic_words:
                    # Calculate the number of keywords from other topics
                    other_topics_count = total_keywords - count
    
                    # Ensure that the number of keywords from other topics does not exceed the limit
                    if other_topics_count <= max_words_other_topics:
                        # Assign this topic as the dominant topic
                        assigned_topics.append((topic_rep_index, topic, self.topic_names[topic]))
                        break
            else:
                # If no valid topic is found, assign None
                assigned_topics.append((topic_rep_index, None))
    
        return assigned_topics
    
    
    #------------------exclusion keywords
    
    """
    NEW: REMOVE SENTENCES WHICH CARRY EXCLUSION KEYWORDS --> e.g. sale topic consists of "market" and "price". If there is a link to stock or fair value, it is not sales-related
        Distinction to Paper 1:
        - Further filtering: e.g. sentences containing 'market' and 'price' are considered to be sales topic. However, often times, they relate to stock or financial instruments. To prevent this, check if the sentence contains any "excluding" keywords, such as "stock", "fair value" or "financial instrument".
    """
    
    def filter_labeled_dataset_by_exclusion(self, labeled_dataset):
        """
        Filters out sentences that contain exclusion keywords for their assigned topic.
    
        Args:
        - labeled_dataset: List of tuples text, (doc_id, topic_int, topic_name), where:
            - sentence (str): The sentence being analyzed.
            - topic_int (int): The assigned topic index.
        - exclusion_dict: Dictionary where keys are topic indices and values are lists of exclusion words.
    
        Returns:
        - A filtered list of (sentence, topic_int) pairs, excluding sentences that contain forbidden keywords.
        """
        filtered_dataset = []
    
        for sentence, (doc_id, topic_int, topic_name) in labeled_dataset:
    
            # Check if there are exclusion words for this topic
            if topic_int in self.exclusion_dict:
                exclusion_words = self.exclusion_dict[topic_int]
    
                # If any exclusion word appears in the sentence, skip it
                if any(exclusion_word in sentence for exclusion_word in exclusion_words):
                    continue  
    
            # Keep the sentence if it passed filtering
            filtered_dataset.append((sentence, (doc_id, topic_int, topic_name)))
            
        print("Filtering of labeled dataset finished.")
        print(f"Number of remaining sentences: {len(filtered_dataset)}")
        print(f"Removed sentences: {len(labeled_dataset) - len(filtered_dataset)}")
    
        return filtered_dataset

    def create_unique_dataset(self, labeled_dataset_texts):
        """
        Create a unique labeled dataset from the full labeled dataset
        """
        #extract indices
        labeled_sentences_indices = [doc_id for text, (doc_id, topic_int, topic_name) in labeled_dataset_texts]
        
        # transform it into sentence, topic_id, doc_id tuples to further filter unique sentences
        foundation_unique_sentences = [(text, topic_id, doc_id) for text, (doc_id, topic_id, topic_name) in labeled_dataset_texts]
        
        #remove the topic name and flatten the list
        labeled_dataset_texts = [(text, topic_int) for text, (doc_id, topic_int, topic_name) in labeled_dataset_texts]
        
        # Use a dictionary to ensure unique sentences
        unique_sentences = {}
        for sentence, topic_id, index in foundation_unique_sentences:
            # Keep the first occurrence or replace based on your requirement
            if sentence not in unique_sentences:  
                unique_sentences[sentence] = (topic_id, index)
        
        # Convert back to a list if needed
        unique_sentence_list = [(sentence, topic_id, index) for sentence, (topic_id, index) in unique_sentences.items()]
        
        labeled_sentences_indices_unique = [index for sent, topic_id, index in unique_sentence_list]
        print(f"Number of labeled sentences (unique): {len(labeled_sentences_indices_unique)}")
        
        # Convert indices list to a set for fast lookup
        labeled_indices_set = set(labeled_sentences_indices)
        labeled_indices_unique_set = set(labeled_sentences_indices_unique)
        

        
        labeled_dataset_texts_unique = [(sentence, topic_id) for (sentence, topic_id, index) in unique_sentence_list] #DO NOT USE THE SET AS IT CONFUSES THE ORDER: list(set(labeled_dataset_new_texts))

        return {
            "labeled_dataset": labeled_dataset_texts, 
            "labeled_dataset_unique": labeled_dataset_texts_unique, 
            "indices": labeled_sentences_indices, 
            "indices_unique": labeled_sentences_indices_unique,
            "keywords": self.keyword_list,
            "topic_names": self.topic_names,
            "blacklist": self.keyword_blacklist,
            "exclusion_dict": self.exclusion_dict
        }

    def run(self, min_topic_words:int = 2, max_words_other_topics:int = 0, exact_match:bool = False):
        """
        Wrapper method to create the labeled dataset
        """
        
        #1. Create keyword-sentence matrix
        keyword_sent_matrix = self.create_keyword_sent_matrix(exact_match = exact_match)
        
        #2. Topic assignments
        topic_assignments = self.analyze_keyword_distribution(keyword_sent_matrix)
        
        #3. create labeled dataset
        labeled_dataset = self.topic_assignment_keywords(topic_assignments, min_topic_words = min_topic_words, max_words_other_topics = max_words_other_topics)
        labeled_dataset_texts = list(zip(self.sentences, labeled_dataset)) #zip with sentences
        labeled_dataset_texts = [tup for tup in labeled_dataset_texts if tup[1][1] is not None] #filter the none-topics

        #4. filter with exclusion dict if given
        if self.exclusion_dict is not None:
            labeled_dataset_texts = self.filter_labeled_dataset_by_exclusion(labeled_dataset_texts)

        #5. prepare unique dataset
        results = self.create_unique_dataset(labeled_dataset_texts)
        
        print(f"Number of labeled sentences: {len(results["labeled_dataset"])}")
        print(f"Number of corresponding indices: {len(results["indices"])}\n")

        
        print(f"Number of unique labeled sentences: {len(results["labeled_dataset_unique"])}")
        print(f"Number of corresponding indices: {len(results["indices_unique"])}")

    
        return results