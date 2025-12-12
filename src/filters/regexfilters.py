import re
class RegexRuleEngine:
    def __init__(self):
        self.suspicious_patterns = [
            r"import\s+os",
            r"import\s+sys",
            r"os\.system\(",
            r"subprocess\.",
            r"exec\(",
            r"eval\(",

            r"<script>",
            r"javascript:",

            r"\/jailbreak",
            r"\[System Mode\]",
            r"ADMIN_Override"
        ]
    def scan(self,text):
        for pattern in self.suspicious_patterns:
            if re.search(pattern,text,re.IGNORECASE):
                return True, f"Regex Match: Detected suspicious pattern '{pattern}'"
        return False,None