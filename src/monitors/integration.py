# Import modules from ALL members
from src.filters import RegexRuleEngine, KeywordFilter, EncodingPatternDetector  # Member 1
from src.analysis import PerplexityAnalyzer, StatisticalAnalyzer, DriftDetector  # Member 2
from src.detection import BertDetector, SemanticDriftCalculator  # Member 3
from .leakage import LeakageMonitor  # Member 4
from .policy import PolicyEnforcer  # Member 4


class SecurePromptPipeline:
    def __init__(self):
        print("Initializing SecurePrompt Pipeline (Weighted Ensemble Mode)...")

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

        # --- CONFIGURATION: Weighted Ensemble ---
        # Adjust these weights based on which module you trust most
        self.weights = {
            "heuristic": 0.2,  # Regex/Keywords (Low trust, high false positives)
            "perplexity": 0.3,  # Gibberish/Obfuscation (Medium trust)
            "bert": 0.5  # Semantic Understanding (High trust)
        }

        # If the weighted sum >= 0.5, the prompt is BLOCKED.
        self.BLOCKING_THRESHOLD = 0.5

    def normalize_perplexity(self, ppl_value):
        """
        Squashes perplexity (0 to infinity) into a 0.0 - 1.0 score.
        Logic: If PPL > 100, we consider it 'fully suspicious' (1.0).
        """
        if ppl_value > 100.0:
            return 1.0
        else:
            return ppl_value / 100.0

    def scan_input(self, user_prompt: str) -> dict:
        """
        Runs the pipeline. If encoding is detected, it decodes the text
        BEFORE sending it to Perplexity and BERT.
        """
        decision = {
            "status": "PASS",
            "reason": "Safe",
            "total_risk": 0.0,
            "breakdown": {}
        }

        # --- 1. Heuristic Layer & Decoding ---
        # We assume self.encoding.scan returns (is_encoded, decoded_text, method_name)
        is_encoded, decoded_text, encoding_method = self.encoding.scan(user_prompt)

        # KEY LOGIC CHANGE:
        # If we successfully decoded it, we analyze the HIDDEN message.
        # If not, we analyze the original user input.
        if is_encoded and decoded_text:
            print(f"[INFO] Decoding Detected ({encoding_method}). Analyzing hidden content...")
            text_to_analyze = decoded_text
            # We still penalize them for trying to hide it!
            score_heuristic = 1.0
        else:
            text_to_analyze = user_prompt

            # Check other heuristics on the original text
            is_keyword_blocked, _ = self.keyword.scan(user_prompt)
            is_regex_sus, _ = self.regex.scan(user_prompt)

            if is_keyword_blocked or is_regex_sus:
                score_heuristic = 1.0
            else:
                score_heuristic = 0.0

        # --- 2. Statistical Analysis (Member 2) ---
        # Analyze the DECODED text (or original if no encoding)
        raw_ppl = self.perplexity.calculate_score(text_to_analyze)
        score_ppl = self.normalize_perplexity(raw_ppl)

        # --- 3. Transformer Detection (Member 3) ---
        # Analyze the DECODED text
        score_bert = self.bert.predict_probability(text_to_analyze)

        # --- 4. Weighted Calculation ---
        total_risk = (
                (score_heuristic * self.weights["heuristic"]) +
                (score_ppl * self.weights["perplexity"]) +
                (score_bert * self.weights["bert"])
        )

        # --- 5. Final Decision ---
        if total_risk >= self.BLOCKING_THRESHOLD:
            decision["status"] = "BLOCK"
            decision[
                "reason"] = f"High Risk ({total_risk:.2f}) - Hidden Intent Detected" if is_encoded else f"High Risk ({total_risk:.2f})"

        decision["total_risk"] = round(total_risk, 4)
        decision["breakdown"] = {
            "heuristic_score": score_heuristic,
            "perplexity_norm": round(score_ppl, 2),
            "bert_prob": round(score_bert, 4),
            "analyzed_content": text_to_analyze[:50] + "..."  # Log what we actually read
        }

        return decision

    def scan_output(self, llm_response: str) -> dict:
        """
        Scans the LLM output for leakage or policy violations.
        (Kept mainly as Member 4 implemented it)
        """
        decision = {"status": "PASS", "reason": None}

        # Check Leakage
        is_leaked, reason = self.leakage.check_output(llm_response)
        if is_leaked:
            decision["status"] = "BLOCK"
            decision["reason"] = reason
            return decision

        # Check Policy
        is_violation, reason = self.policy.validate_response(llm_response)
        if is_violation:
            decision["status"] = "BLOCK"
            decision["reason"] = reason
            return decision

        return decision