import pickle
import os
from labeled_dataset.utils_labeled_dataset import keywords, topic_names
from rag.RAGPreparation import RAGPreparator
from rag.utils_rag import system_prompt, scope_topics



os.chdir("/mnt/sdb1/home/simonj")

result_loc = "paper2/Results/RAG/prompts.pkl"
rel_model = "acl_modern_bert"


#-------------------------------------Load relevant data
#------Texts and metadata
text_loc = "paper2/Data/Text/item7_text_rel_tickers_quantile.pkl"
with open(text_loc, "rb") as file:
    data = pickle.load(file)

texts = data["item7_texts"]
metadata = data["item7_metadata"]
print(f"Number of texts: {len(texts)}")
print(f"Number of metadata: {len(metadata)}")


#-----Cluster properties
loc = "paper2/Results/knn/knn_cluster_properties_fd_sp500_quantile.pkl"

with open(loc, "rb") as file:
    cluster_props = pickle.load(file)

entropies = cluster_props[rel_model]["entropies"]
s_scores = cluster_props[rel_model]["silhouette_scores"].get()
#intratopic_sims = cluster_props[rel_model]["intratopic_sims"]
#mean_intratopic_sim = cluster_props[rel_model]["mean_intratopic_sim"]
#intertopic_sim = cluster_props[rel_model]["intertopic_sim"]

print(f"Number of silhouette scores: {len(s_scores)}")
print(f"Number of entropies: {len(entropies)}")


#-------full embeddings
embedding_loc = "paper2/Data/embeddings/embeddings_fd_sp500_quantile.pkl"

with open(embedding_loc, "rb") as file:
    data = pickle.load(file)

embeddings = data[rel_model]
print(f"Shape of embeddings: {embeddings.shape}")


#-------labeled dataset embeddings
labeled_embeddings_loc = "paper2/Data/embeddings/labeled_dataset_embeddings_fd.pkl"

with open(labeled_embeddings_loc, "rb") as file:
    data = pickle.load(file)

labeled_embeddings = data[rel_model]
print(f"Shape of labeled embeddings: {labeled_embeddings.shape}")

#-----labeled dataset
labeled_dataset_loc = "paper2/Data/add_labeled_dataset/full_labeled_dataset.pkl"

with open(labeled_dataset_loc, "rb") as file:
    data = pickle.load(file)

labeled_dataset = data["labeled_dataset"]
print(f"Number of instances in labeled dataset: {len(labeled_dataset)}")

labeled_sentences = [sent for sent, topic in labeled_dataset]
labeled_topics = [topic for sent, topic in labeled_dataset]
print(f"Number of labeled sentences: {len(labeled_sentences)}")
print(f"Number of labeled topics: {len(labeled_topics)}")

#Generate prompts
prep = RAGPreparator(
    s_scores = s_scores,
    entropies = entropies,
    embeddings = embeddings, 
    sentences = texts, 
    metadata = metadata, 
    topic_names = topic_names, 
    keywords = keywords, 
    labeled_embeddings = labeled_embeddings, 
    labeled_sentences = labeled_sentences, 
    labeled_topics = labeled_topics, 
    labeled_explanations = None,
)
results = prep.run_prompt_generation(
    s_score_threshold = 0.5, 
    entropy_threshold = 0.5, 
    max_margin_prev_next_sentence = 2, 
    n_neighbors = 2, 
    metric = "cosine", 
    debug = False
)

with open(result_loc, "wb") as file:
    pickle.dump(results, file)

print(f"Results saved to {result_loc}")