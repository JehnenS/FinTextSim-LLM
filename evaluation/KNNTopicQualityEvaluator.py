import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from bertopic.vectorizers import ClassTfidfTransformer


class KNNTopicQualityEvaluator:
    """
    Evaluate the quality of the KNN topics - ctfidf based assignment of words
    """
    def __init__(
        self,
        documents,
        labels,
        keyword_list,
        topic_names,
        ):
        
        self.documents = documents
        self.labels = labels.flatten()
        self.keyword_list = keyword_list
        self.topic_names = topic_names

    def _create_topic_df_(self):
        """
        Function to create a data frame which consists of the topics (first column) and the concatenated documents per topic --> foundation for calculating ctf-idf scores
        """
        
        #create dictionary of topic label and corresponding sentences
        label_dict = {}
    
        for sent, label in zip(self.documents, self.labels):
            if label == -1: #skip noise data points
                continue
                
            #check if label is already in dictionary
            if label not in label_dict:
                #add label if not present - initialize with empty list
                label_dict[label] = []
    
            #append sentence to corresponding labels list
            label_dict[label].append(sent)
    
        #sort the dictionary --> ascending from 0 to n_topics
        label_dict = dict(sorted(label_dict.items()))
    
        #combine sentences for each label into a single document
        combined_docs = {label: " ".join(sent) for label, sent in label_dict.items()}
    
        #create df for the combined documents
        df = pd.DataFrame(combined_docs.items(), columns = ["label", "document"])
        print(f"Shape of df: {df.shape}")
        return df
    
    def _vectorize_(self, df, vectorizer):
        """
        Vectorize the df of topics and concatenated documents per topic to obtain a dtm
        """
        #1. create dtm
        dtm = vectorizer.fit_transform(df["document"])
        print(f"Shape of dtm: {dtm.shape}")
    
        #2. Extract tokens from vectorizer and find keyword-token matches
        tokens = vectorizer.get_feature_names_out()
        flat_keyword_list = [word for sublist in self.keyword_list for word in sublist] #flatten the keyword list
        print(f"Number of keywords: {len(flat_keyword_list)}")
    
        #one-hot encode if a token is in the keyword list --> substring approach
        one_hot_token_keyword = [1 if any(keyword in token.lower() for keyword in flat_keyword_list) else 0 for token in tokens]
        
        print(f"Number of tokens: {len(one_hot_token_keyword)}")
        print(f"Number of matches: {one_hot_token_keyword.count(1)}")
        print(f"Number of other tokens: {one_hot_token_keyword.count(0)}")
    
        #extract the token names from the one-hot encoded list
        one_hot_tokens = [token for one_hot, token in zip(one_hot_token_keyword, tokens) if one_hot == 1]
        print(f"Number of tokens: {len(one_hot_tokens)}")
        
        return dtm, one_hot_tokens, one_hot_token_keyword
    
    
    
    
    def _weigh_tokens_(self,dtm, seed_columns, bm25_weighting = True, reduce_frequent_words = True, seed_multiplier = 1):
        """
        weight the tokens from the dtm with ctfidf
        Optionally: multiply seed_columns with a multiplier to increase their effect
        """
        # Create the weighting model
        weight_model = ClassTfidfTransformer(bm25_weighting=bm25_weighting, reduce_frequent_words=reduce_frequent_words)
        ctfidf_matrix = weight_model.fit_transform(dtm)
        print(f"Shape of ctfidf_matrix: {ctfidf_matrix.shape}")
    
        # Make a copy of the matrix to apply the weights without modifying the original
        ctfidf_matrix_weighted = ctfidf_matrix.copy()
    
        if seed_multiplier != 1:
            # Iterate over each column and apply the multiplier if seed_columns[i] == 1
            for i in range(ctfidf_matrix.shape[1]):
                if seed_columns[i] == 1:
                    ctfidf_matrix_weighted[:, i] *= seed_multiplier
    
        print(f"Shape of ctfidf_matrix_weighted: {ctfidf_matrix_weighted.shape}")
        
        return ctfidf_matrix_weighted
    
    
    def _create_ctfidf_matrix_(self, vectorizer, bm25_weighting = True, reduce_frequent_words = True, seed_multiplier = 1):
        """
        Wrapper function to create a ctfidf matrix
        Step 1: Create dataframe of topics and concatenated topic documents
        Step 2: Create a document-term-matrix with topics as documents
        Step 3: Weight the word counts with ctfidf --> importance of words across topics
        """
        #1. create the df
        df = self._create_topic_df_()
    
        #2. create dtm and extract the seed words (one-hot-tokens)
        dtm, seed_words, seed_columns = self._vectorize_(df, vectorizer)
    
        #3. create the ctfidf-matrix
        ctfidf_matrix = self._weigh_tokens_(dtm, seed_columns, bm25_weighting = bm25_weighting, reduce_frequent_words = reduce_frequent_words, seed_multiplier = seed_multiplier)
        
        return ctfidf_matrix
    
    
    def _extract_top_tokens_per_row_(self, ctfidf_matrix, vectorizer, top_n=5):
        """
        Extracts the top tokens with the highest c-TF-IDF scores per row.
    
        Arguments:
            ctfidf_matrix: A 2D numpy array or sparse matrix of c-TF-IDF scores.
            feature_names: A list of terms corresponding to the columns of the ctfidf_matrix.
            top_n: The number of top tokens to extract per row.
    
        Returns:
            A list of lists containing top tokens for each row.
        """
        #extract feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Convert ctfidf_matrix to dense format if it's sparse
        ctfidf_matrix = ctfidf_matrix.toarray() if not isinstance(ctfidf_matrix, np.ndarray) else ctfidf_matrix
    
        # Initialize a list to store top tokens for each row
        top_tokens_per_row = []
    
        # Iterate over each row in the ctfidf_matrix
        for row in ctfidf_matrix:
            # Get indices of top N scores in the row
            top_indices = np.argsort(row)[-top_n:][::-1]  # Sort and get top indices in descending order
    
            # Get corresponding tokens and scores
            top_tokens = [(feature_names[i], row[i]) for i in top_indices]
    
            # Store the tokens and their scores
            top_tokens_per_row.append(top_tokens)
    
        return top_tokens_per_row
    
    
    
    def _plot_scores_(self, subplot_data, plots_per_row=5, file_loc=None):
        """
        Plot ctfidf scores per topic
        """
    
        num_subplots = len(subplot_data)
        num_rows = (num_subplots + plots_per_row - 1) // plots_per_row
        fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(9, 1.8 * num_rows))
        axes = axes.flatten() if num_rows > 1 else [axes]
    
        for i, (ax, data) in enumerate(zip(axes, subplot_data)):
            words, scores = zip(*data)
            ax.barh(words, scores, color='skyblue', edgecolor='black')
            title = f'Topic {i} - {self.topic_names[i]}' if self.topic_names is not None else f'Topic {i}'
            ax.set_title(title, fontsize=16)
            ax.set_xlabel('ctfidf Scores', fontsize=14)
            #ax.set_ylabel('Words', fontsize=10)
            ax.tick_params(axis='x', labelsize=14) 
            ax.tick_params(axis='y', labelsize=14)
            ax.invert_yaxis()
    
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
    
        try:
            plt.tight_layout()
        except Exception as e:
            print(f"Warning: tight_layout() failed: {e}")
    
        if file_loc is not None:
            plt.savefig(file_loc, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
    
    
    
    def run_ctfidf_eval(self,  vectorizer, plots_per_row = 5, file_loc = None):
        """
        combined function to obtain the top tokens and the score plots
        """
        ctfidf_matrix = self._create_ctfidf_matrix_(vectorizer = vectorizer)
        top_tokens = self._extract_top_tokens_per_row_(ctfidf_matrix, vectorizer)
        self._plot_scores_(top_tokens, plots_per_row, file_loc)
        
        return top_tokens


    #----------------keyword heatmap
    
    def _extract_dtm_keywords_(self, vectorizer):
        """
        Create a document term matrix that lists the occurrences of the keywords per topic.
        Currently: only exact matches of the keywords.
        """
        # Create topic data frame
        df = self._create_topic_df_()
    
        # Create document-term-matrix
        dtm, _, _ = self._vectorize_(df, vectorizer)
    
        # Flatten keyword list if it is nested
        all_keywords = [kw for sublist in self.keyword_list for kw in sublist] \
                       if any(isinstance(el, list) for el in self.keyword_list) \
                       else self.keyword_list
    
        # Store the index of the keywords in the dtm vocabulary
        vocab = vectorizer.get_feature_names_out()
        keyword_indices_dtm = []
        for keyword in all_keywords:
            index = np.where(vocab == keyword)[0].flatten()
            if index.size > 0:
                keyword_indices_dtm.append((keyword, index.item()))
            else:
                keyword_indices_dtm.append((keyword, None))
    
        # Extract valid keywords and their indices
        keyword_words = [word for word, col_index in keyword_indices_dtm if col_index is not None]
        keyword_indices = [col_index for _, col_index in keyword_indices_dtm if col_index is not None]
    
        # Slice the dtm
        dtm_keywords = dtm[:, keyword_indices]
        print(f"Shape of keyword-dtm: {dtm_keywords.shape}")
    
        return dtm_keywords, keyword_words
    
    
    def plot_keyword_heatmap(self, dtm_keywords, keyword_words, topic_keywords=None, return_df=False):
        """
        Plot a heatmap of the keyword frequency per document.
        """
        dtm_keywords_df = dtm_keywords.toarray()
        scaler = MinMaxScaler()
        dtm_keywords_normalized = scaler.fit_transform(dtm_keywords_df)
    
    
        df = pd.DataFrame(dtm_keywords_normalized, columns=keyword_words, index=self.topic_names)
    
        plt.figure(figsize=(12, 8))
        sns.heatmap(df, cmap="YlGnBu", cbar=True, annot=False, linewidths=0.5, linecolor='black')
    
        # Optional: add red borders for topic-specific keywords
        if topic_keywords:
            for topic_keywords_list in topic_keywords:
                for kw in topic_keywords_list:
                    if kw in keyword_words:
                        idx = keyword_words.index(kw)
                        plt.gca().add_patch(
                            plt.Rectangle((idx, 0), 1, dtm_keywords.shape[0], fill=False, edgecolor="red", lw=2)
                        )
    
        plt.title("Keyword Heatmap", fontsize=16)
        plt.xlabel("Keywords", fontsize=12)
        plt.ylabel("Documents", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.show()
    
        if return_df:
            return df
    
    
    def run_keyword_heatmap(self, topic_keywords=None, vectorizer=None, return_df=False):
        dtm_keywords_matrix, dtm_keywords_words = self._extract_dtm_keywords_(vectorizer)
        return self.plot_keyword_heatmap(dtm_keywords_matrix, dtm_keywords_words, topic_keywords, return_df=return_df)