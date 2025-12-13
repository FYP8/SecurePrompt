import torch
import os
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification


class BertDetector:
    def __init__(self, model_path='models/bert_classifier'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not os.path.exists(model_path):
            print(f"Warning: Local model not found at {model_path}. Using 'bert-base-uncased'.")
            model_path = 'bert-base-uncased'
        else:
            print(f"Loading BERT Classifier from {model_path}...")

        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def predict_probability(self, text):
        """
        Returns a float between 0.0 (Safe) and 1.0 (Malicious).
        Required for the Weighted Ensemble voting system.
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        # Move to device (GPU/CPU)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            # Apply Softmax to get probabilities (0.0 - 1.0)
            probs = torch.softmax(logits, dim=1)

            # We assume Index 1 = "Malicious" (Check your training labels if unsure!)
            malicious_score = probs[0][1].item()

        return malicious_score

    def predict(self, text: str):
        """
        Returns: (is_malicious (bool), confidence_score (float))
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)

        # Assuming Index 1 is Malicious (based on standard labeling logic)
        malicious_score = probs[0][1].item()
        is_malicious = malicious_score > 0.5

        return is_malicious, malicious_score