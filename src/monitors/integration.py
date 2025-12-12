# Import modules from ALL members
from src.filters import RegexRuleEngine, KeywordFilter, EncodingPatternDetector  # Member 1
from src.analysis import PerplexityAnalyzer, StatisticalAnalyzer, DriftDetector  # Member 2
from src.detection import BertDetector, SemanticDriftCalculator  # Member 3
from .leakage import LeakageMonitor  # Member 4
from .policy import PolicyEnforcer  # Member 4


class SecurePromptPipeline:
    def __init__(self):
        print("Initializing SecurePrompt Pipeline...")

        # --- Layer 1: Heuristics (Member 1) ---
        self.regex = RegexRuleEngine()
        self.keyword = KeywordFilter()
        self.encoding = EncodingPatternDetector()

        # --- Layer 2: Analysis (Member 2) ---
        self.perplexity = PerplexityAnalyzer()  # Loads DistilGPT2
        self.stats = StatisticalAnalyzer()
        self.drift = DriftDetector()

        # --- Layer 3: Deep Detection (Member 3) ---
        self.bert = BertDetector()  # Loads BERT

        # --- Layer 4: Output Monitoring (Member 4) ---
        self.leakage = LeakageMonitor()
        self.policy = PolicyEnforcer()

        # Thresholds (Configurable)
        self.PPL_THRESHOLD = 100.0  # Max perplexity allowed
        self.BERT_CONFIDENCE = 0.80  # Min confidence to auto-block

    def scan_input(self, user_prompt: str) -> dict:
        """
        Runs the full input validation pipeline.
        Returns a decision dict.
        """
        decision = {
            "status": "PASS",
            "reason": None,
            "metrics": {}
        }

        # 1. Heuristic Scan (Fast Fail)
        # -----------------------------
        is_encoded, _, reason = self.encoding.scan(user_prompt)
        if is_encoded: return self._block(decision, reason)

        is_blocked, kw = self.keyword.scan(user_prompt)
        if is_blocked: return self._block(decision, f"Keyword Block: {kw}")

        is_sus, reason = self.regex.scan(user_prompt)
        if is_sus: return self._block(decision, reason)

        # 2. Statistical Analysis
        # -----------------------------
        ppl_score = self.perplexity.calculate_score(user_prompt)
        entropy = self.stats.calculate_entropy(user_prompt)

        decision["metrics"]["perplexity"] = round(ppl_score, 2)
        decision["metrics"]["entropy"] = round(entropy, 2)

        if ppl_score > self.PPL_THRESHOLD:
            # Just a warning for now, or block if very high
            decision["warnings"] = "High Perplexity (Possible Obfuscation)"

        # 3. Transformer Detection
        # -----------------------------
        is_malicious, confidence = self.bert.predict(user_prompt)
        decision["metrics"]["bert_malicious_prob"] = round(confidence, 4)

        if is_malicious and confidence > self.BERT_CONFIDENCE:
            return self._block(decision, f"BERT Detected Attack (Conf: {confidence:.2%})")

        return decision

    def scan_output(self, llm_response: str) -> dict:
        """
        Scans the LLM output for leakage or policy violations.
        """
        decision = {"status": "PASS", "reason": None}

        # Check Leakage
        is_leaked, reason = self.leakage.check_output(llm_response)
        if is_leaked:
            return self._block(decision, reason)

        # Check Policy
        is_violation, reason = self.policy.validate_response(llm_response)
        if is_violation:
            return self._block(decision, reason)

        return decision

    def _block(self, decision, reason):
        decision["status"] = "BLOCK"
        decision["reason"] = reason
        return decision