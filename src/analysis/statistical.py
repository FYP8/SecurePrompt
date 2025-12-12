import math
from collections import Counter


class StatisticalAnalyzer:
    def calculate_entropy(self, text: str) -> float:
        """
        Calculates Shannon Entropy.
        High entropy (>4.5) often indicates encrypted or random strings.
        """
        if not text:
            return 0.0

        counts = Counter(text)
        total_len = len(text)
        entropy = 0.0

        for count in counts.values():
            p_x = count / total_len
            if p_x > 0:
                entropy += -p_x * math.log2(p_x)

        return entropy

    def get_token_metrics(self, text: str) -> dict:
        return {
            "entropy": self.calculate_entropy(text),
            "length": len(text),
            "word_count": len(text.split())
        }