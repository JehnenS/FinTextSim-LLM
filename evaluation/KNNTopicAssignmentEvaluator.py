import pickle
import numpy as np
import cupy as cp
from cuml.metrics.cluster.silhouette_score import cython_silhouette_samples as cu_silhouette_score
from sklearn.model_selection import StratifiedKFold
from cuml.metrics import pairwise_distances
from tqdm import tqdm

class KNNTopicAssignmentEvaluator:
    """
    Evaluate the structural properties of the generated clusters
    """
    def __init__(self, knn_loc, embedding_loc = None):
        self.knn_loc = knn_loc
        self.embedding_loc = embedding_loc


    def _load_knn_results_(self, model_name):
        """
        Load embeddings (based on Paper 1)
        """
        with open(self.knn_loc, "rb") as file:
            data = pickle.load(file)

        self.distances = data[model_name]["distances"]
        self.indices = data[model_name]["indices"]
        print(f"KNN results loaded")

    def _load_embeddings_(self, model_name):
        """
        Load embeddings (based on Paper 1)
        """
        with open(self.embedding_loc, "rb") as file:
            data = pickle.load(file)

        self.embeddings = data[model_name]
        print(f"Embeddings loaded - Shape: {self.embeddings.shape}")


    def create_topic_embeddings(self, embeddings, topics):
        """
        Function to create topic embeddings by taking the mean of all embeddings belonging to that specific topic
    
        Returns a matrix with embeddings per topic: dimensions: num_topics x model dimensions
        """
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
    
    #-------------------------evaluation
    def calculate_intertopic_cosine_similarity(self):
        """
        Accelerated calculation of intertopic cosine similarity using cuML.
        Returns:
            cosine_sim_matrix_cpu: numpy matrix of cosine similarity between topics
            mean_upper_triangle: mean cosine similarity of the upper triangle of the similarity matrix
        """
        #1. generate topic embeddings and move to GPU
        topics = np.array(self.indices[:,0].flatten())
        topic_embeddings = self.create_topic_embeddings(self.embeddings, topics) 
        topic_embeddings_gpu = cp.asarray(topic_embeddings)

        #2. compute cosine similarity with cuml
        cosine_sim_matrix = 1 - pairwise_distances(topic_embeddings_gpu,
                                                   topic_embeddings_gpu,
                                                   metric='cosine')

        #3. bring similarit matrix back to CPU
        cosine_sim_matrix_cpu = cp.asnumpy(cosine_sim_matrix)

        #4. extract upper triangle without diagonal
        upper_triangle_indices = np.triu_indices(cosine_sim_matrix_cpu.shape[0], k=1)
        upper_triangle_values = cosine_sim_matrix_cpu[upper_triangle_indices]

        #5. extract mean similarity
        mean_upper_triangle = np.mean(upper_triangle_values)

        print(f"Mean intertopic similarity: {mean_upper_triangle:.3f}")
        return cosine_sim_matrix_cpu, mean_upper_triangle


    def calculate_intratopic_cosine_similarity_batch(self, batch_size: int = 100000):
        """
        Calculate mean cosine similarity between sentence embeddings of each topic
        and the corresponding topic embedding, using GPU with batching.
        """
    
        cosine_similarities = []
        topics = np.array(self.indices[:, 0].flatten())
        unique_topics = np.unique(topics)
    
        # Create topic embeddings (CPU, then move to GPU later in loop)
        topic_embeddings = self.create_topic_embeddings(self.embeddings, topics)
    
        embeddings_cpu = self.embeddings  # keep embeddings on CPU to avoid huge GPU allocations
    
        for idx, topic in tqdm(enumerate(unique_topics), desc="Topic Progress Intratopic Similarity", total=len(unique_topics)):
            # mask on CPU
            topic_mask = (topics == topic)
            rel_embeddings_cpu = embeddings_cpu[topic_mask]
    
            # topic embedding to GPU
            topic_embedding_gpu = cp.asarray(topic_embeddings[idx].reshape(1, -1))
    
            # collect sims for this topic
            sims_topic = []
    
            # process in batches
            for start in range(0, rel_embeddings_cpu.shape[0], batch_size):
                end = start + batch_size
                batch_cpu = rel_embeddings_cpu[start:end]
    
                # move batch to GPU
                batch_gpu = cp.asarray(batch_cpu)
    
                # cosine similarity (1 - cosine distance)
                sims = 1 - pairwise_distances(
                    batch_gpu, topic_embedding_gpu, metric="cosine"
                )
                sims_topic.append(cp.asnumpy(sims[:, 0]))
    
                # free GPU memory
                del batch_gpu, sims
                cp._default_memory_pool.free_all_blocks()
    
            cosine_similarities.append(np.concatenate(sims_topic))
    
        # mean similarity per topic
        mean_intratopic_sim_per_topic = [np.mean(arr) for arr in cosine_similarities]
        mean_intratopic_sim = np.mean(mean_intratopic_sim_per_topic)
    
        print(f"Mean intratopic similarity: {mean_intratopic_sim:.3f}")
        return cosine_similarities, mean_intratopic_sim

    

    def compute_silhouette_scores_in_batches(self, n_splits=4, random_state=42):
        """
        Compute silhouette scores for large CuPy-based embedding matrix using batches --> stratified to have similar distribution of topics per batch
    
        Parameters:
            X_cupy (cp.ndarray): [n_samples, n_features] embedding matrix (on GPU).
            labels (array-like): [n_samples,] integer labels corresponding to each row.
            n_splits (int): Number of batches to split data into.
            random_state (int): Random seed for reproducibility.
    
        Returns:
            silhouette_scores (cp.ndarray): [n_samples,] silhouette score per row, in original order.
        """
        labels = np.array(self.indices[:,0].flatten())  #ensure labels are NumPy for sklearn
        X_cupy = cp.asarray(self.embeddings)
        n_samples = X_cupy.shape[0]
        silhouette_scores = cp.zeros(n_samples, dtype=cp.float32)  #preallocate output on GPU
    
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        X_dummy = np.zeros(n_samples)  # Placeholder, not used in stratification
    
        for _, batch_idx in tqdm(skf.split(X_dummy, labels), desc = "Split Progress Silhouette Scores", total = n_splits):
            batch_idx = cp.asarray(batch_idx)  #move indices to GPU
            X_batch = X_cupy[batch_idx] #extract batch embedding matrix
            labels_batch = cp.asarray(labels[batch_idx.get()])  # Ensure on GPU
    
            #compute silhouette scores for the batch
            batch_scores = cu_silhouette_score(X_batch, labels_batch)
            print(f"Silhouette score batch: {np.mean(batch_scores.get()):.3f}")
            # Assign back to full array
            silhouette_scores[batch_idx] = batch_scores
    
        return silhouette_scores


    def compute_entropy(self, temperature = 10):
        """
        Calculate the entropy of datapoints --> based on distance of datapoint to the topic embeddings
        take the softmax probabilities
        first softmax probability: probability for closest neighbor/nearest topic embedding
        """
        # Compute softmax over negative distances
        exp_neg_dist = np.exp(-self.distances * temperature)
        softmax_probs = exp_neg_dist / exp_neg_dist.sum(axis=1, keepdims=True)
        
        #print(softmax_probs.shape)  # Should match distances.shape (num_rows, num_neighbors)
        flat_probs = softmax_probs[:, 0].flatten() #probability for closest neighbor
    
        #softmax_probs is a (num_rows, num_neighbors) matrix
        entropy = -np.sum(softmax_probs * np.log(softmax_probs + 1e-8), axis=1)
    
        #print(entropy.shape)  # Should be (num_rows,)
        print(f"Entropies calculated - {entropy.shape}")
        return entropy

    def run(self, model_name, temperature:int = 10, n_splits:int = 4, random_state:int = 42, batch_size = 100000):
        """
        Wrapper method to run evaluation
        """
        self._load_knn_results_(model_name)
        self._load_embeddings_(model_name)
        
        print("Begin evaluation.")
        entropies = self.compute_entropy(temperature = temperature)
        silhouette_scores = self.compute_silhouette_scores_in_batches(n_splits = n_splits, random_state = random_state)
        intratopic_sims, mean_intratopic_sim = self.calculate_intratopic_cosine_similarity_batch(batch_size = batch_size)
        intertopic_sim = self.calculate_intertopic_cosine_similarity()

        result_dict = {
            "entropies": entropies,
            "silhouette_scores": silhouette_scores,
            "intratopic_sims": intratopic_sims,
            "mean_intratopic_sim": mean_intratopic_sim,
            "intertopic_sim": intertopic_sim
        }

        # cleanup: free embeddings and knn data
        del self.embeddings, self.distances, self.indices
        import gc
        gc.collect()
        cp._default_memory_pool.free_all_blocks()
        
        return result_dict