import pickle
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset



class ModalityExtractor:
    """
    Extract modalities from sentences
    """
    def __init__(self, 
                 text_loc:str = "paper1/Data/topic_model_input/bertopic_input_2016_2023.pkl"):
        
        self.text_loc = text_loc
        self.texts = None #initialize as None


    def _load_texts_(self, text_name = "texts"):
        """
        Load Item 7 texts
        """
        with open(self.text_loc, "rb") as file:
            data = pickle.load(file)

        self.texts = data[text_name]


    def sentiment_determination(self, model, tokenizer, batch_size=2056, device='cuda', text_name = "texts"):
        """
        Batch-optimized sentiment analysis function.
    
        Inputs:
        - texts: list of text strings
        - model: HuggingFace model
        - tokenizer: HuggingFace tokenizer
        - batch_size: batch size for processing
        - device: 'cuda' or 'cpu'
    
        Output:
        - list of tuples: (sentiment label id, probability)
        """
        model.to(device)
        #model.eval()

        # Make sure texts are loaded
        if self.texts is None:
            self._load_texts_(text_name = text_name)
        
        results = []

        dataset = TextDataset(self.texts)
        dataloader = DataLoader(dataset, batch_size=batch_size)
    
        for batch in tqdm(dataloader, desc="Batch Processing"):
            # Tokenize the batch
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            
            # Inference
            with torch.no_grad():
                outputs = model(**inputs)
    
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
    
            # Process batch results
            pred_class_ids = torch.argmax(probs, dim=1)
            pred_probs = probs.gather(1, pred_class_ids.unsqueeze(1)).squeeze(1)
    
            batch_results = list(zip(pred_class_ids.tolist(), pred_probs.tolist()))
            results.extend(batch_results)
    
        return results

    def classify_fls(self, model, tokenizer, batch_size=2048, device='cuda', text_name = "texts"):
        """
        Classify forward-looking statements in batches for speed.
        
        Returns:
        - list of tuples: (label index, probability)
        """
        model.to(device)
        
        # Make sure texts are loaded
        if self.texts is None:
            self._load_texts_(text_name = text_name)
            
        results = []

        dataset = TextDataset(self.texts)
        dataloader = DataLoader(dataset, batch_size=batch_size)

    
        for batch in tqdm(dataloader, desc="Classifying FLS"):
            # Tokenize
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
    
            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs)
    
            logits = outputs.logits
            probs = F.softmax(logits, dim=1) #transform logits into softmax probabilities
            
            # Process batch results
            pred_class_ids = torch.argmax(probs, dim=1)
            pred_probs = probs.gather(1, pred_class_ids.unsqueeze(1)).squeeze(1)
    
            batch_results = list(zip(pred_class_ids.tolist(), pred_probs.tolist()))
            results.extend(batch_results)
    
        return results



class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]
