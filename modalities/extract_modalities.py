import pickle
import os
os.chdir("/mnt/sdb1/home/simonj") #set wd


result_loc = "paper2/Data/Modalities/modalities_fd_sp500_quantile.pkl"
text_loc = "paper2/Data/Text/item7_text_rel_tickers_quantile.pkl"

#-------------Import sentiment model
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"

# Load the tokenizer
tokenizer_sentiment = AutoTokenizer.from_pretrained(model_name)

# Load the model
distilroberta_fin = AutoModelForSequenceClassification.from_pretrained(model_name)


#-----------Import FLS model
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-fls',num_labels=3)
tokenizer_fls = BertTokenizer.from_pretrained('yiyanghkust/finbert-fls')

#-------------Extract modalities
from modalities.ModalityExtractor import ModalityExtractor

extractor = ModalityExtractor(
    text_loc = text_loc
)

sentiment_results = extractor.sentiment_determination(
    model = distilroberta_fin,
    tokenizer = tokenizer_sentiment,
    batch_size = 2048,
    text_name = "item7_texts"
)

fls_results = extractor.classify_fls(
    model = finbert,
    tokenizer = tokenizer_fls,
    batch_size = 2048,
    text_name = "item7_texts"
)


results = {
    "sentiment": sentiment_results,
    "fls": fls_results,
    "text_loc": text_loc
}

with open(result_loc, "wb") as file:
    pickle.dump(results, file)

print(f"Results saved to {result_loc}")