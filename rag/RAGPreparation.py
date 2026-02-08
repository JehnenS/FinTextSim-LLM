import pandas as pd
import ollama
import numpy as np
import pickle
from cuml.neighbors import NearestNeighbors
import cupy as cp
from pydantic import BaseModel, Field
from typing import List
import torch
from tqdm import tqdm
import json

from labeled_dataset.utils_labeled_dataset import topic_names as topic_taxonomy
topic_taxonomy.append("None")  # add None to avoid forcibly assigning sentences to topics



class RAGPreparator:
    def __init__(self, s_scores, entropies, embeddings, sentences, metadata, topic_names, keywords, labeled_embeddings, labeled_sentences, labeled_topics, labeled_explanations = None):
        """
        class to perform the RAG evaluation on the critical sentences 
        """
        self.s_scores = s_scores
        self.entropies = entropies
        self.embeddings = embeddings
        self.sentences = sentences
        self.metadata = metadata
        self.topic_names = topic_names
        self.keywords = keywords
        self.labeled_embeddings = labeled_embeddings
        self.labeled_sentences = labeled_sentences
        self.labeled_topics = labeled_topics
        self.labeled_explanations = labeled_explanations

    #------------------Filter the data based on silhouette scores and entropy
    def _identify_sentences_indices_to_check(self, s_score_threshold = 0.5, entropy_threshold = 0.5):
        """
        Get the sentence id's of the ones which need to be checked --> entropy > 0.5 & ssc < 0.5
        s_scores: list/np.array of silhouette scores for each datapoint corresponding to the row in embedding matrix
        entropies: list/np.array of entropies for each datapoint corresponding to the row in embedding matrix
        s_score_threshold: s_scores below threshold are marked as critical
        entropy_threshold: entropies above threshold are marked as critical
        """
    
        # Find critical points based on both entropy and silhouette score
        critical_scores = [(i, s_score, ent) for i, (s_score, ent) in enumerate(zip(self.s_scores, self.entropies)) 
                         if (s_score < s_score_threshold) and (ent > entropy_threshold)]
    
        indices_to_check = [i for (i, s_score, ent) in critical_scores]
        print(f"Number of indices to check: {len(indices_to_check)}")
        print(f"Share of indices to check: {len(indices_to_check) / len(self.s_scores) * 100:.2f}%")
        return indices_to_check
    
    def _determine_data_to_check(self, indices_to_check):
        """
        Determine the data which will be put into the RAG approach
        embeddings: embedding matrix
        sentences: list of sentences corresponding to rows in embedding matrix
        metadata: list of metadata corresponding to rows in embedding matrix
        """
        embeddings_to_check = self.embeddings[indices_to_check]
        print(f"Shape of embedding matrix to check: {embeddings_to_check.shape}\n")
    
        sentences_to_check = [self.sentences[i] for i in tqdm(indices_to_check, desc="Sentence Filtering")]
        print(f"Number of sentences: {len(sentences_to_check)}\n")
        
        metadata_to_check = [self.metadata[i] for i in tqdm(indices_to_check, desc="Metadata Filtering")]
        print(f"Number of metadata: {len(metadata_to_check)}\n")
        
        return embeddings_to_check, sentences_to_check, metadata_to_check

    #-----------------------generate data relevant for filling in the prompt


    #---------------Metadata string
    def _format_metadata_string(self, metadata_to_check):
        """
        Format a string with metadata for the sentence
    
        {'period_of_report': '20231201',
         'year_of_report': '2023',
         'cik': '0000796343',
         'company_name': 'ADOBE INC.\r',
         'filing_date': '20240117',
         'change_date': '20240116',
         'doc_id': 0,
         'sentence_id': 0}
    
        Returns a list of all metadata strings for the sentences to check
        """
    
        all_metadata_strings = []
        for meta in metadata_to_check:
            company_name = meta.get("company_name").rstrip('\r')
            #cik = meta[4]
            #gics_sector = meta[8]
    
            metadata_str = (
                f"Company: {company_name}\n"
             #   f"Sector: {gics_sector}"
            )
            all_metadata_strings.append(metadata_str)
    
        print(f"Number of metadata strings: {len(all_metadata_strings)}")
            
        return all_metadata_strings

        
    #------------Context string
    def _identify_context_sentences(self, indices_to_check, sentences_to_check, metadata_to_check, max_margin_prev_next_sentence = 2, debug = False):
        """
        Identify the context of the input sentences
        Guarantee that the context sentences are from the same document --> overlap to other documents does not make sense
    
        max_margin_prev_next_sentence: limit the next and previous sentence to ensure that context is prevailed --> e.g. if 20 sentences in between target and previous are deleted due to number of words, context is not guaranteed
    
        """
        context_sentences = []
        for i, (sentence, index) in enumerate(zip(sentences_to_check, indices_to_check)):
            #get target sentence and its metadata
            target_metadata = metadata_to_check[i]
            target_doc_id = target_metadata.get("doc_id")  # Extract document ID
            target_sentence_id = target_metadata.get("sentence_id")  # Extract sentence ID (last element)
            target_index = indices_to_check[i]  # Get the initial index of the sentence
            
            if debug:
                print(f"Target sentence: {sentence}\n")
                print(f"Target metadata: {target_metadata}\n")
                print(f"Target index: {target_index}")
    
            # Initialize previous and next sentences
            prev_sentence, next_sentence = None, None
    
            # Check if there's a valid previous sentence in the same document
            if target_index > 0:  # Ensure index is not negative
                prev_metadata = self.metadata[target_index - 1]
                prev_doc_id = prev_metadata.get("doc_id")  # Document ID of previous sentence
                prev_sentence_id = prev_metadata.get("sentence_id")  # Sentence ID of previous sentence (last element)
                #print(f"Previous metadata: {prev_metadata}")
                
                if (prev_doc_id == target_doc_id) and (target_sentence_id - prev_sentence_id <= max_margin_prev_next_sentence):  # Ensure it belongs to the same document and is within rasonable sentence range
                    prev_sentence = self.sentences[target_index - 1]
                
            # Check if there's a valid next sentence in the same document
            if target_index < len(self.sentences) - 1:  # Ensure index is in range
                next_metadata = self.metadata[target_index + 1]
                next_doc_id = next_metadata.get("doc_id") # Document ID of next sentence
                next_sentence_id = next_metadata.get("sentence_id") # Sentence ID of next sentence (last element)
                #print(f"Next metadata: {next_metadata}")
                
                if (next_doc_id == target_doc_id) and (next_sentence_id - target_sentence_id <= max_margin_prev_next_sentence):  # Ensure it belongs to the same document and is within rasonable sentence range
                    next_sentence = self.sentences[target_index + 1]
    
            context_sentences.append((prev_sentence, sentence, next_sentence))
    
        print(f"Number of context sentences: {len(context_sentences)}")
    
        return context_sentences

    def _format_context_string(self, prev_current_next_sentence_list):
        """
        Given a list of (prev, current, next) sentence tuples, return a list of formatted context strings.
        """
        context_strings = []
    
        for prev, current, next_sent in prev_current_next_sentence_list:
            lines = []
            if prev:
                lines.append(f"Previous Sentence: {prev}")
            if next_sent:
                lines.append(f"Next Sentence: {next_sent}")
    
            context_strings.append("\n".join(lines))
    
        return context_strings

    #-------------knn examples

    def _knn_labeled_dataset(self, embeddings_to_check, n_neighbors = 5, metric = "cosine"):
        """
        Determine the nearest neighbors for the embeddings to check based on the labeled dataset
        """
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)  
    
        # Fit KNN with the topic embeddings
        knn.fit(self.labeled_embeddings, self.labeled_topics)
    
        # Find the nearest topics for each of the embeddings to check
        distances, indices = knn.kneighbors(embeddings_to_check)
        print("KNN finished")
    
        return distances, indices
    
    def _format_knn_example_string(self, knn_indices):
        """
        Format an example string from the semantically closest neighbors of the sentence
        
        knn_indices: np.array/list of the indices --> result from KNN (e.g., [5, 12, 33])
        labeled_topics: np.array/list of topics from the labeled dataset where KNN is fit on
        labeled_sentences: np.array/list of sentences from the labeled dataset
        labeled_explanations: list of LLM explanations generated for the labeled dataset
        """
        all_example_strings = []
    
        for neighbors in knn_indices:
            example_parts = []
            for i, n in enumerate(neighbors):
                text = f"**Example {i+1}:**\n"
                text += f"**Text:** \"{self.labeled_sentences[n]}\"\n"
                text += f"**Assigned Topic:** {self.topic_names[self.labeled_topics[n]]}\n"
                if self.labeled_explanations is not None:
                    text += f"**Reasoning:** {self.labeled_explanations[n]}"
                example_parts.append(text)
            
            example_str = "\n\n".join(example_parts)
            all_example_strings.append(example_str)
    
        return all_example_strings



    def generate_prompts(self, texts, metadata_str, context_strs, knn_examples):
        """
        Efficient prompt generation using precomputed string blocks and f-strings.
        Assumes:
          - context_strs: list of context strings (one per text)
          - knn_examples: list of KNN example strings (one per text)
        """
         
        prompts = [
            f"Classify the following sentence into the given label taxonomy, strictly following the classification rules and response format provided in the system instructions above:\n"
            f"Sentence: {text}\n\n"
            f"### Context:\n{context}\n\n"
            f"### Examples of semantically closest labeled neighbors:\n{knn}\n\n"
            #f"### Guidelines:\n"
            #f"- estimate the **confidence** in the sentence being correctly assigned to the topic"
            #f"- provide a percentage probability (0-100%)"
            for text, knn, context in tqdm(zip(texts, knn_examples, context_strs), total=len(texts), desc="Prompt generation")
        ]
    
        return prompts


    def generate_prompts(
        self,
        texts: list[str],
        context_strs: list[str],
        knn_examples: list[str],
        metadata_strs: list[str],
        include_guidelines: bool = True,
        include_context: bool = True,
        include_examples: bool = True,
        include_metadata:bool = True
    ):
        """
        Lightweight aligned prompt generation.
        No misalignment between text, context, and KNN examples.
        """

        guideline_block = (
            "Classify the following sentence into the given label taxonomy, strictly following the classification rules and response format provided in the system instructions.\n"
            if include_guidelines else ""
        )

        prompts = []
        for i, (text, context, knn, meta) in enumerate(
            tqdm(zip(texts, context_strs, knn_examples, metadata_strs), total=len(texts), desc="Prompt generation")
        ):
            parts = []

            # 1. Optional short guideline header
            if guideline_block:
                parts.append(guideline_block.rstrip())

            # 2. The actual classification target
            parts.append(f"### Sentence:\n{text}")

            # 3. Optional context block
            if include_context:
                parts.append(f"### Prev/Next Context:\n{context}")

            # 4. Optional KNN examples block
            if include_examples:
                parts.append(f"### KNN Neighbors:\n{knn}")

            # 5. Optional metadata (only if you end up needing it)
            if include_metadata:
                parts.append(f"### Metadata:\n{meta}")

            # Join with intentional blank line separators to preserve boundaries
            prompt_final = "\n\n".join(parts)
            prompts.append(prompt_final)

        return prompts




    def run_prompt_generation(self, s_score_threshold = 0.5, entropy_threshold = 0.5, max_margin_prev_next_sentence = 2, n_neighbors = 2, metric = "cosine", debug = False):
        """
        Wrapper method to perform the generation of prompts
        """
        #1. Get the indices of the sentences which do not meet the silhouette/entropy criteria
        indices_to_check = self._identify_sentences_indices_to_check(s_score_threshold = s_score_threshold, entropy_threshold = entropy_threshold)

        #2. Extract the relevant data based on indices_to_check
        embeddings_to_check, sentences_to_check, metadata_to_check = self._determine_data_to_check(indices_to_check)

        #3. Generate/format metadata strings
        metadata_strings = self._format_metadata_string(metadata_to_check)

        #4. Identify context sentences and generate/format context strings
        context_sentences = self._identify_context_sentences(indices_to_check, sentences_to_check, metadata_to_check, max_margin_prev_next_sentence = max_margin_prev_next_sentence, debug = debug)
        context_strings = self._format_context_string(context_sentences)

        #5. Fit knn on labeled dataset and generate/format example strings
        distances, indices = self._knn_labeled_dataset(embeddings_to_check, n_neighbors = n_neighbors, metric = metric)
        knn_example_strings = self._format_knn_example_string(indices)


        #7. Generate prompts
        prompts = self.generate_prompts(sentences_to_check, context_strings, knn_example_strings, metadata_to_check)
        prompts_minimal = self.generate_prompts(sentences_to_check, context_strings, knn_example_strings, metadata_to_check, include_examples = False, include_guidelines = False, include_metadata = False)

        if debug:
            print(f"Examples metadata strings: \n {metadata_strings[:5]}")
            print("\n")
            print(f"Examples context sentences: \n {context_sentences[:5]}")
            print("\n")
            print(f"Examples formatted context strings: \n {context_strings[:5]}")
            print("\n")
            print(f"Examples formatted knn-strings: \n {knn_example_strings[:5]}")
            print("\n")
            print(f"Topic string:\n {topic_str}")
            print("\n")
            for prompt in prompts[:5]:
                print(prompt)

        return {
            "prompts": prompts, 
            "prompts_minimal": prompts_minimal, 
            "indices_to_check": indices_to_check, 
            "sentences_to_check": sentences_to_check
        }