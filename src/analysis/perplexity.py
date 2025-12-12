import torch
import os
import math
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


class PerplexityAnalyzer:
    def __init__(self, model_path='models/distilgpt2_finetuned'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Check if local model exists, otherwise verify path
        if not os.path.exists(model_path):
            print(f"Warning: Local model not found at {model_path}. Using base 'distilgpt2'.")
            model_path = 'distilgpt2'
        else:
            print(f"Loading Fine-Tuned GPT from {model_path}...")

        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def calculate_score(self, text: str) -> float:
        """
        Calculates Perplexity (PPL). Lower = More natural. Higher = Anomalous.
        Formula: exp(CrossEntropyLoss)
        """
        if not text or len(text.strip()) == 0:
            return 0.0

        encodings = self.tokenizer(text, return_tensors='pt').to(self.device)
        input_ids = encodings.input_ids

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss

        return math.exp(loss.item())