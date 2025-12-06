#--------------------------lemmatization, keep relevant parts of speech
#spacy
import nltk
from nltk.tokenize import sent_tokenize
import re


def replace_subs(text, substitutions_list):
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

import re

def clean_sentence(sentence):
    # Remove punctuation using regex (only keeps relevant context)
    sentence = re.sub(r'[^\w\s]', '', sentence)
    # Remove extra whitespace at the beginning and end
    sentence = sentence.strip()
    return sentence




#-------------------------clean the sentences --> punctuation, whitespaces

def subs_sentence_lengths_filter(texts, metadata, substitutions, min_words = 5, max_words = 50):
    """
    Combined function to replace substitutions and filter sentences by sentence length for both metadata and text data
    """
    #Replace substitutions
    texts_subs = [replace_subs(text, substitutions) for text in tqdm(texts, desc = "Replace substitutions")]
    print(f"Substitutions replaced.\nNumber of documens: {len(texts_subs)}")

    #filter sentences based on length
    texts_sent_length = [(i, sent) for (i, sent) in tqdm(enumerate(texts_subs), desc = "Filtering sentences by word count") if min_words < len(sent.split()) < max_words]
    print(f"Number of relevant sentences after filtering by number of words (min: {min_words}; max: {max_words}): {len(texts_sent_length)}")
    print(f"Filtered sentences: {len(texts_subs) - len(texts_sent_length)}")

    sentences = [sent for i, sent in texts_sent_length] #sentences to keep after filtering for sentence length
    print(f"Number of sentences to keep: {len(sentences)}")
    
    indices = [i for i, sent in texts_sent_length] #indices to keep after filtering for sentence length --> corresponding to sentences
    print(f"Number of indices to keep: {len(indices)}")

    #clean whitespaces and punctuation
    sentences_final = [clean_sentence(sent) for sent in tqdm(sentences, desc = "Cleaning punctuation and whitespaces")]
    print(f"Number of documents: {len(sentences_final)}; Sentences cleaned (punctuation, whitespaces after sentence-length-based filter")

    #------------------Filter metadata
    indices_set = set(indices)  # Convert list to set for faster lookup

    metadata_final = [meta for i, meta in tqdm(enumerate(metadata), desc="Metadata filtering") if i in indices_set]

    print(f"Remaining metadata entries: {len(metadata_final)}")
    return sentences_final, metadata_final
    