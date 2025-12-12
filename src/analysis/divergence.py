import random
import numpy as np
from scipy.special import rel_entr


class DivergenceAnalyzer:
    def __init__(self, mutation_rate=0.1):
        self.mutation_rate = mutation_rate

    def mutate_input(self, text: str) -> str:
        """
        Applies JAILGUARD-style random mutations (Swap/Delete) to the input.
        Ref: Base Paper, Section 4.2.1 [cite: 363-366]
        """
        chars = list(text)
        num_mutations = max(1, int(len(chars) * self.mutation_rate))

        for _ in range(num_mutations):
            if len(chars) < 2: break
            idx = random.randint(0, len(chars) - 1)
            op = random.choice(['swap', 'delete'])

            if op == 'swap':
                swap_idx = (idx + 1) % len(chars)
                chars[idx], chars[swap_idx] = chars[swap_idx], chars[idx]
            elif op == 'delete' and len(chars) > 5:
                chars.pop(idx)

        return "".join(chars)

    def calculate_kl_divergence(self, probs_p, probs_q) -> float:
        """
        Calculates Kullback-Leibler Divergence between two probability distributions.
        Ref: Base Paper Eq (3) [cite: 233-234]
        """
        # Add epsilon to prevent log(0) errors
        epsilon = 1e-10
        p = np.array(probs_p) + epsilon
        q = np.array(probs_q) + epsilon

        # Normalize
        p /= p.sum()
        q /= q.sum()

        return sum(rel_entr(p, q))