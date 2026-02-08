import pandas as pd
from tqdm import tqdm
import re

class TextCleaner:
    def __init__(self, texts):
        """
        Clean the text from additional dataset
        """
        self.texts = texts

    def replace_subs(self, text, substitutions_list):
        """
        function to replace substitutions in text (words + re patterns) --> before tokenizing, etc.
    
        text: document
        substitutions list: list of tuples containing substitutions and replacements
        """
        text = text.lower() #convert to lower case
    
        text_subbed = text
        for pattern, replacement in substitutions_list:
            text_subbed = re.sub(pattern, replacement, text_subbed)
    
        # Ensure single whitespaces
        text_subbed = re.sub(r'\s+', ' ', text_subbed).strip()
    
        return text_subbed



    def clean_sentence(self, text):
        # Remove punctuation using regex (only keeps relevant context)
        text_clean = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespace at the beginning and end
        text_clean = text_clean.strip()
        return text_clean
    
    
    
    
    #-------------------------clean the sentences --> punctuation, whitespaces
    
    def run(self, substitutions, min_words = 5, max_words = 50):
        """
        Combined function to replace substitutions and filter sentences by sentence length for both metadata and text data
        """
        #Replace substitutions
        texts_subs = [self.replace_subs(text, substitutions) for text in tqdm(self.texts, desc = "Replace substitutions")]
        print(f"Substitutions replaced.\nNumber of documens: {len(texts_subs)}")
    
        #filter sentences based on length
        texts_sent_length = [(i, sent) for (i, sent) in tqdm(enumerate(texts_subs), desc = "Filtering sentences by word count") if min_words < len(sent.split()) < max_words]
        print(f"Number of relevant sentences after filtering by number of words (min: {min_words}; max: {max_words}): {len(texts_sent_length)}")
        print(f"Filtered sentences: {len(texts_subs) - len(texts_sent_length)}")
    
        sentences = [sent for i, sent in texts_sent_length] #sentences to keep after filtering for sentence length
        print(f"Number of sentences to keep: {len(sentences)}")
        
        
        #clean whitespaces and punctuation
        sentences_final = [self.clean_sentence(sent) for sent in tqdm(sentences, desc = "Cleaning punctuation and whitespaces")]
        print(f"Number of documents: {len(sentences_final)}; Sentences cleaned (punctuation, whitespaces after sentence-length-based filter")
    
        
        return sentences_final