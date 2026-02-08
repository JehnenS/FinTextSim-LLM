import os
import pickle
from tqdm import tqdm
os.chdir("/mnt/sdb1/home/simonj")

from evaluation.KNNTopicAssignmentEvaluator import KNNTopicAssignmentEvaluator


model_names = [
    #"acl_bert",
    #"acl_finbert",
    "acl_modern_bert",
]

#define loc to save the results
save_loc = "paper2/Results/knn/knn_cluster_properties_fd_sp500_quantile.pkl"


#-----------Define evaluator
evaluator = KNNTopicAssignmentEvaluator(
    embedding_loc = "paper2/Data/embeddings/embeddings_fd_sp500_quantile.pkl",
    knn_loc = "paper2/Results/knn/knn_centroid_fd_sp500_quantile.pkl",
)

#---------iterate over all models
all_results = {} # create dictionary to store results

#iterate over all models
for model_name in tqdm(model_names, desc = "Model Progress"):
    print(f"Model: {model_name}")
    results = evaluator.run(model_name = model_name, n_splits = 20, batch_size = 100000)
    all_results[model_name] = results
    print(f"Results for {model_name} appended to full results dictionary.\n")


#----------Save results
with open(save_loc, "wb") as file:
    pickle.dump(all_results, file)

print(f"Results saved to {save_loc}")