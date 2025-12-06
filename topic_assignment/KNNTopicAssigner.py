import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
from cuml.neighbors import NearestNeighbors
import cupy as cp
from cuml.metrics.cluster.silhouette_score import cython_silhouette_samples as cu_silhouette_score
from sklearn.model_selection import StratifiedKFold
from cuml.metrics import pairwise_distances



class KNNTopicAssigner:
    """
    KNN-based topic assigner (pseudo-topic assignment)
    """
    def __init__(self, labeled_dataset_loc, embedding_loc = None, embedding_loc_batches = None):
        self.embedding_loc = embedding_loc
        self.labeled_dataset_loc = labeled_dataset_loc
        self.embedding_loc_batches = embedding_loc_batches

        self.embeddings = None #initialize as None
        self.labeled_embeddings = None
        self.labeled_topics = None
        self.distances = None
        self.indices = None

    def _load_embeddings(self, model_name):
        """
        Load embeddings (based on Paper 1)
        """
        with open(self.embedding_loc, "rb") as file:
            data = pickle.load(file)

        self.embeddings = data[model_name]
        print(f"Embeddings loaded - Shape: {self.embeddings.shape}")

    def _load_labeled_embeddings_(self, model_name):
        """
        Load labeled dataset
        """
        with open(self.labeled_dataset_loc, "rb") as file:
            data = pickle.load(file)

        self.labeled_topics = data["basics"]["topics"]
        self.labeled_embeddings = data[model_name]
        print(f"Labeled embeddings loaded - Shape: {self.labeled_embeddings.shape}")

    def _load_embeddings_batches(self):
        """
        Load batches of embeddings and combine them
        """
        import glob
        
        #----------merge all batch embeddings
        # Path pattern where batch files were saved
        batch_files = sorted(glob.glob(self.embedding_loc_batches))
        
        all_embeddings = []
        
        for f in tqdm(batch_files, desc = "Load batch embeddings and concatenate them vertically"):
            with open(f, "rb") as handle:
                batch = pickle.load(handle)
                all_embeddings.append(batch)
        
        # Combine into one big array
        self.embeddings = np.vstack(all_embeddings)
        print(f"Final embeddings shape: {self.embeddings.shape}")
        


    def create_topic_embeddings(self, embeddings, topics):
        """
        Function to create topic embeddings by taking the mean of all embeddings belonging to that specific topic
    
        Returns a matrix with embeddings per topic: dimensions: num_topics x model dimensions
        """
        print("Create topic embeddings")
        unique_topics = np.unique(topics)
        topic_embeddings = []
       
    
        for topic in unique_topics:
            # Extract embeddings for the current topic
            topic_mask = topics == topic
            rel_embeddings = embeddings[topic_mask]
            topic_embeddings.append(np.mean(rel_embeddings, axis=0))
            
        topic_embeddings = np.array(topic_embeddings)
        print(f"Shape of topic embedding matrix: {topic_embeddings.shape}")
        return topic_embeddings


    def fit_knn(self, topic_embeddings):
        """
        fit knn model
        """
        print("Fit KNN model")
        n_neighbors = topic_embeddings.shape[0] #align with number of topics
        
        # Initialize cuML KNN
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')  #
        
        # Fit KNN with the topic embeddings
        knn.fit(topic_embeddings)
        
        # Find the nearest topics for each embedding in your data
        distances, indices = knn.kneighbors(self.embeddings)

        self.distances = distances
        self.indices = indices
        print("KNN fitted.")

    def fit_knn_batches(self, topic_embeddings, batch_size=1000000):
        """
        Fit the knn approach in batches to avoid running into memory issures
        """
        #create lists to store all results
        all_distances, all_indices = [], []

        #define the number of neighbors
        n_neighbors = topic_embeddings.shape[0] #align with number of topics
        
        #initialize cuML KNN
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')

        #fit knn with topic embeddings
        knn.fit(topic_embeddings)
        
        for start in tqdm(range(0, self.embeddings.shape[0], batch_size), desc = "Batch progress KNN"):
            batch = self.embeddings[start:start+batch_size] #extract batch embeddings
            distances, indices = knn.kneighbors(batch) #calculate batch distances and indices
            all_distances.append(distances) #append the distances and indices to the list
            all_indices.append(indices)

        print("KNN fitted")

        self.distances = np.vstack(all_distances)
        self.indices = np.vstack(all_indices)


    def run(self, model_name):
        """
        Wrapper method to assign pseudo-labels
        """
        #load full embeddings and labeled embeddings
        self._load_embeddings(model_name)
        self._load_labeled_embeddings_(model_name)

        #create labeled topic embeddings
        topic_embeddings = self.create_topic_embeddings(self.labeled_embeddings, self.labeled_topics)

        #fit knn
        self.fit_knn(topic_embeddings)   

        result_dict = {
            "distances": self.distances,
            "indices": self.indices,
        }

        
        return result_dict

    def run_batch_embeddings(self, model_name):
        """
        Wrapper method to assign pseudo labels based on batched embeddings
        """
        self._load_embeddings_batches()
        self._load_labeled_embeddings_(model_name)

        #create labeled topic embeddings
        topic_embeddings = self.create_topic_embeddings(self.labeled_embeddings, self.labeled_topics)

        #fit knn
        self.fit_knn_batches(topic_embeddings)   

        result_dict = {
            "distances": self.distances,
            "indices": self.indices,
        }

        return result_dict