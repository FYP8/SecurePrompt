import torch


class EmbeddingExtractor:
    def __init__(self, detector_instance):
        # We reuse the loaded model from BertDetector to save RAM
        self.tokenizer = detector_instance.tokenizer
        self.model = detector_instance.model.bert  # Access the base transformer
        self.device = detector_instance.device

    def get_embedding(self, text: str):
        """
        Extracts the [CLS] token embedding as the sentence vector.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Last hidden state: [batch_size, seq_len, hidden_dim]
            # We take the first token [CLS] for classification representation
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return cls_embedding