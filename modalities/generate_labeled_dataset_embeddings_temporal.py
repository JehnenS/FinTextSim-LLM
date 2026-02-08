import pickle
import os

os.chdir("/mnt/sdb1/home/simonj")

save_loc = "paper2/Data/embeddings/labeled_dataset_embeddings_fd_temporal.pkl"

#----------import test and train sets
file_loc = "paper2/Data/FinTextSim/train_test_sets/train_test_sets_full_dataset_temporal.pkl"

with open(file_loc, "rb") as file:
    data = pickle.load(file)

texts = data["labeled_sentences"]
topics = data["labeled_dataset_topics"]



print(f"Number of texts: {len(texts)}")

#create dictionary to store results
results = {}

results["basics"] = {
    "topics": topics
}


#-------------define model locs and names
from sentence_transformers import SentenceTransformer

model_locs = [
    #"paper2/Fintextsim_Models/fintextsim_acl_fd_BERT",
    #"paper2/Fintextsim_Models/fintextsim_acl_fd_finbert",
    "paper2/Fintextsim_Models/fintextsim_acl_fd_temporal_modern_bert"
]

model_names = [
    #"acl_bert",
    #"acl_finbert",
    "acl_modern_bert"
]


#-------------------Create embeddings


#iterate over each model-loc
for i, model_loc in enumerate(model_locs):
    #load sentence transformer
    model = SentenceTransformer(model_loc)

    print(f"Sentence-Transformer loaded - {model_names[i]}")
    embeddings = model.encode(texts, show_progress_bar = True)

    #save results with model name as key
    results[model_names[i]] = embeddings


#----------Save results
with open(save_loc, "wb") as file:
    pickle.dump(results, file)

print(f"Results saved to {save_loc}")