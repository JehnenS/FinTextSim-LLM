from tqdm import tqdm
import pickle
import os
os.chdir("/mnt/sdb1/home/simonj")
from topic_assignment.KNNTopicAssigner import KNNTopicAssigner


model_names = [
    #"acl_bert",
    #"acl_finbert",
    "acl_modern_bert",
]

#define loc to save the results
save_loc = "paper2/Results/knn/knn_centroid_fd_sp500_quantile.pkl"

#---------------Define assigner
assigner = KNNTopicAssigner(
    embedding_loc = "paper2/Data/embeddings/embeddings_fd_sp500_quantile.pkl",
    labeled_dataset_loc = "paper2/Data/embeddings/labeled_dataset_embeddings_fd.pkl",
)

#-------------Run the assignment
#create dictionary to store results
all_results = {}

#iterate over all models
for model_name in tqdm(model_names, desc = "Model Progress"):
    results = assigner.run(model_name = model_name)

    #append results
    all_results[model_name] = results
    print(f"Results for {model_name} appended to full results dictionary.")

#----------Save results
with open(save_loc, "wb") as file:
    pickle.dump(all_results, file)

print(f"Results saved to {save_loc}")