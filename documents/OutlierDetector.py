import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm
import statistics
import matplotlib.pyplot as plt



class OutlierDetector:
    def __init__(self,
                 text_list:list,
                 metadata_list:list,
                ):

        self.text_list = text_list
        self.metadata_list = metadata_list

    def filter_none_text(self):
        """
        method to check item extraction. Keep only those, where the text is not None
        names of documents (indices), None values due to non-parsing, checking text length, etc.
        input: split documents list
        output: list of filtered texts and list of filtered metadata corresponding to the filtered texts
        """
     
        #1. filter docs where the text is not None
        filtered_text_metadata_list = [(text, meta) for i, (text, meta) in tqdm(enumerate(zip(self.text_list, self.metadata_list)), desc = "Filter by None text") if text is not None]
        filtered_text_list, filtered_metadata_list = zip(*filtered_text_metadata_list)
        
        print(f"docs at beginning: {len(self.text_list)}")
        print(f"successful parses: {len(filtered_text_list)}")
        print(f"share of succesful parses: {round(len(filtered_text_list)/len(self.text_list)*100, 2)}%")
    
        
        print(f"docs at beginning: {len(self.text_list)}")
        print(f"docs filtered by None text: {len(self.text_list) - len(filtered_text_list)}")
        print(f"docs remaining after removing none text: {len(filtered_text_list)}")

        self.text_list = filtered_text_list
        self.metadata_list = filtered_metadata_list
        
        return self
    
    
    def filter_min_words(self, min_words_threshold):
        """
        Function to filter documents that lie below a minimum words threshold.
        
        input: output from filter_none_text_function: list of lists (original document number, text, text_length)
        filter list by minimum threshold of chars
        output: list of list, filtered by None Text and min_chars (original document number, text, no_of_words)
        """
        #1.get number of words for each text
        number_of_words = []
        
        #iterate over each text
        for text in self.text_list:
            words = text.split() #split text into words (based on whitespace) and store in list
            number_of_words.append(len(words)) #len of list equals number of words in document
    
        #plt.boxplot(number_of_words)
        #plt.title("Boxplot #words - pre-filter")
        #plt.show()
    
        #filter text and metadata based on the min_words_threshold
        filtered_text_metadata_list = [(text, meta) for text, meta, num_words in tqdm(zip(self.text_list, self.metadata_list, number_of_words), desc = "Filter by min. words", total = len(self.text_list)) if num_words >= min_words_threshold]
        filtered_text_list, filtered_metadata_list = zip(*filtered_text_metadata_list)
     
        print(f"docs at beginning/before removing by min_chars: {len(self.text_list)}")
        print(f"docs filtered by min_words (n = {min_words_threshold}): {len(self.text_list) - len(filtered_text_list)}")
        print(f"docs remaining after removing min_words (n = {min_words_threshold}): {len(filtered_text_list)}")
    
        #extract number of words post-filter
        #1. Get number of words for each text
        number_of_words_post = []
        
        #iterate over each text to get the mean number of words for the filtered texts
        for text in tqdm(filtered_text_list, desc = "Count number of words per document"):
            words = text.split() #split text into words (based on whitespace) and store in list
            number_of_words_post.append(len(words)) #len of list equals number of words in document
            
        #plt.boxplot(number_of_words_post)
        #plt.title("Boxplot #words - post-filter")
        #plt.show()

        self.text_list = filtered_text_list
        self.metadata_list = filtered_metadata_list
    
        return self, number_of_words_post
    
    def get_zscores(self, number_of_words):
        """
        Function to determine the z-score of each document.
        
        input: list containing the number of words per text
        calculate z-scores of text length
        output list of z scores corresponding to text
        """
        #1. calculate main statistics after applying min_number_words filter
        mean = statistics.mean(number_of_words) 
        sd = statistics.stdev(number_of_words)
    
        print(f"mean #words: {mean}")
        print(f"sd #words: {sd}")
    
        #3. calculate z score for each element in the list
        z_scores = [abs((item - mean) / sd) for item in number_of_words]
        
        #plt.boxplot(z_scores)
        #plt.title("Boxplot z-scores unfiltered")
        #plt.show()
    
        print(f"Mean z-score: {np.array(z_scores).mean():.2f}")
        print(f"sd z-score: {np.array(z_scores).std():.2f}")
        print(f"Max. z-score: {np.array(z_scores).max():.2f}")
    
        return z_scores
    
    def filter_zscores(self, z_scores, z_score_threshold):
        """
        Function to filter documents based on the z-score.
        
        input: list of texts; list of metadata and list of z-scores
        filter list based on z-score threshold
        output: two filtered list of lists: 1: list of just the text that is filtered by zscore and 2: list of lists (original doc number, text, num_of_words, z-score)
        """
        #1. filter elements based on z_score threshold
        filtered_text_metadata_z_score_list = [(text, meta, z_score) for text, meta, z_score in tqdm(zip(self.text_list, self.metadata_list, z_scores), desc = "z-score filtering") if z_score <= z_score_threshold]
        filtered_text_list, filtered_metadata_list, filtered_z_score_list = zip(*filtered_text_metadata_z_score_list)
        
        #print boxplot of filtered z-scores
        #plt.boxplot(filtered_z_score_list)
        #plt.title("Boxplot - z_score_filtered")
        #plt.show()
    
        print(f"docs before filtering by z-scores: {len(self.text_list)}")
        print(f"docs filterd by zscores: {len(self.text_list) - len(filtered_text_list)}")
        print(f"docs after filtering by z-scores: {len(filtered_text_list)}")
    
        print(f"Mean z-score: {np.array(filtered_z_score_list).mean():.2f}")
        print(f"sd z-score: {np.array(filtered_z_score_list).std():.2f}")
        print(f"Max. z-score: {np.array(filtered_z_score_list).max():.2f}")
    
    
        #2. extract text lengths and print boxplot
        number_of_words = []
        for text in filtered_text_list:
            num_words = len(text.split())
            number_of_words.append(num_words)
        
        #plt.boxplot(number_of_words)
        #plt.title("Boxplot - final text lengths")
        #plt.show()
    
        mean = statistics.mean(number_of_words) 
        sd = statistics.stdev(number_of_words)
        min_val = min(number_of_words)
    
        print(f"mean text length: {mean}")
        print(f"sd text length: {sd}")
        print(f"min text length: {min_val}")

        self.text_list = filtered_text_list
        self.metadata_list = filtered_metadata_list
        
        return self

    def filter_by_quantile(self, num_words, quantile_threshold = 0.99):
        """
        Function to filter documents based on the quantile values of number of words.
            
        input: list of texts; list of metadata and list of number of words per document
        filter list based on number of words quantile threshold
        output: two filtered list of lists: 1: list of just the text that is filtered by quantile and 2: list of lists (original doc number, text, num_of_words, z-score)
        """
    
        #1. Determine the value at quantile edge
        quantile_value = np.quantile(num_words, q = quantile_threshold)
        
        #1. filter elements based on number of words
        filtered_text_metadata_list = [(text, meta, n_words) for text, meta, n_words in tqdm(zip(self.text_list, self.metadata_list, num_words), desc = f" {quantile_threshold}% Quantile filtering", total = len(self.text_list)) if n_words <= quantile_value]
        filtered_text_list, filtered_metadata_list, filtered_num_words_list = zip(*filtered_text_metadata_list) 
        
        print(f"docs before filtering by quantiles: {len(self.text_list)}")
        print(f"docs filterd by quantiles: {len(self.text_list) - len(filtered_text_list)}")
        print(f"docs after filtering by quantiles: {len(filtered_text_list)}")
        
        
        fig, axes = plt.subplots(1, 2, figsize = (16,4))
        axes[0].boxplot(filtered_num_words_list)
        axes[0].set_title("Boxplot - final text lengths")
        axes[0].grid(alpha = 0.2)
        
        axes[1].hist(filtered_num_words_list, edgecolor = "white", bins = 100)
        axes[1].set_title("Histogram - final text lengths")
        axes[1].grid(alpha = 0.2)
    
        fig.suptitle("Distribution of number of words after filtering")
        fig.tight_layout()
        fig.show()
        
        mean = np.mean(filtered_num_words_list) 
        sd = np.std(filtered_num_words_list)
        min_val = min(filtered_num_words_list)
        
        print(f"mean text length: {mean}")
        print(f"sd text length: {sd}")
        print(f"min text length: {min_val}")

        self.text_list = filtered_text_list
        self.metadata_list = filtered_metadata_list
    
        return self

    def save_results(self, texts, metadata, output_dir):
        """
        Save results to output directory
        """
        result_dict = {
            "item7_texts": texts,
            "item7_metadata": metadata
        }

        with open(output_dir, "wb") as file:
            pickle.dump(result_dict, file)

        print(f"Texts and metadata saved to {output_dir}")
    
    def run_z_score(self, min_words_threshold:int = 250, z_score_threshold:float = 2.0, output_dir:str = "paper2/Data/Text/10-K/item7/item7_text_outlier.pkl"):
        """
        Combined function for outlier detection --> keep just relevant and comparable documents
        Steps:
        1. Filter documents that have no text
        2. Filter out documents with lesser number of words than min_words_threshold
        3. Calculate z-scores of the remaining documents
        4. Filter out documents with z_score greater than z_score_threshold
    
        Input:
        text_list: list of texts
        min_words_threshold (int): min. threshold for number of words in document --> if num_words < threshold: document discarded
        z_score_threshold (float): max. threshold for z_score of document --> if z_score doc > z_score_threshold: document discarded
        """
        #1. Filter documents that have no text
        self.filter_none_text()
        
        #2. Filter documents with number of words below threshold
        _, number_of_words_post = self.filter_min_words(min_words_threshold)
    
        #3. Create zscores of the filtered lists text lengths
        z_scores = self.get_zscores(number_of_words_post)
    
        #4. Filter based on the z_score_threshold
        self.filter_zscores(z_scores, z_score_threshold)

        #5. Save results
        self.save_results(self.text_list, self.metadata_list, output_dir)

    def run_quantile(self, min_words_threshold:int = 250, quantile_threshold:float = 2.0, output_dir:str = "paper2/Data/10-K/item7/item7_text_outlier.pkl"):
        """
        Combined function for outlier detection --> keep just relevant and comparable documents
        Steps:
        1. Filter documents that have no text
        2. Filter out documents with lesser number of words than min_words_threshold
        3. Calculate critical quantile for number of words
        4. Filter out documents with words > quantile value
    
        Input:
        text_list: list of texts
        min_words_threshold (int): min. threshold for number of words in document --> if num_words < threshold: document discarded
        z_score_threshold (float): quantile for # words --> if # words > quantile: document discarded
        """
        #1. Filter documents that have no text
        self.filter_none_text()
        
        #2. Filter documents with number of words below threshold
        _, number_of_words_post = self.filter_min_words(min_words_threshold)
    
    
        #3. Filter based on the z_score_threshold
        self.filter_by_quantile(number_of_words_post, quantile_threshold)

        #4. Save results
        self.save_results(self.text_list, self.metadata_list, output_dir)