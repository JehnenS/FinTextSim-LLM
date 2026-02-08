import pickle
import os
from labeled_dataset.utils_labeled_dataset import keywords, topic_names
from RAGEvaluator import RAGEvaluator
from utils_rag import system_prompt, scope_topics
import asyncio
import ollama
from tqdm.asyncio import tqdm as async_tqdm


os.chdir("/mnt/sdb1/home/simonj")

result_loc = "paper2/Results/RAG/rag_fd_sp500_quantile_temporal.pkl"
rel_model = "acl_modern_bert"


#-------------------------------------Load relevant data
prompt_loc = "paper2/Results/RAG/prompts_temporal.pkl"

with open(prompt_loc, "rb") as file:
    data = pickle.load(file)

print(f"Keys in the data: {data.keys()}")

prompts = data["prompts"]
prompts_minimal = data["prompts_minimal"]
indices_answers = data["indices_to_check"]

print(f"Number of prompts: {len(prompts)}")
print(f"Number of indices: {len(indices_answers)}")

prompts = prompts_minimal
print(f"Prompt example:")
print(prompts[0])


#----------load texts
text_loc = "paper2/Data/Text/item7_text_rel_tickers_quantile.pkl"
with open(text_loc, "rb") as file:
    data = pickle.load(file)

texts = data["item7_texts"]
metadata = data["item7_metadata"]
print(f"Number of texts: {len(texts)}")
print(f"Number of metadata: {len(metadata)}")

#-----------Run RAG approach
#model = "mistral:7b-instruct"
model = "financial-classifier-mistral"

evaluator = RAGEvaluator(
    prompts = prompts,
    sentences = texts,
    indices_to_check = indices_answers,
    output_path = "paper2/Results/RAG/rag_temporal_out"
)


results = evaluator.run_parallel(model, max_workers = 16)


results_dict = {
    "results": results,
    "system_prompt": system_prompt,
    "model": model,
    "fintextsim": rel_model
}

with open(result_loc, "wb") as file:
    pickle.dump(results_dict, file)

print(f"Results saved to {result_loc}")