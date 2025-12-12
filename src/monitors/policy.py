class PolicyEnforcer:
    def __init__(self):
        # Report Ref: "Output Policy Enforcer" [cite: 1666]
        # These are words the LLM should NEVER output in a secured environment.
        self.banned_output_keywords = [
            "I can generate a bomb",
            "I can help you hack",
            "bypass security",
            "illegally",
            "unfiltered response",
            "DAN mode enabled"
        ]

    def validate_response(self, text: str) -> tuple:
        """
        Checks if the response violates safety policies.
        Returns: (is_violation (bool), reason (str))
        """
        text_lower = text.lower()

        for phrase in self.banned_output_keywords:
            if phrase.lower() in text_lower:
                return True, f"Policy Violation: Response contained banned phrase '{phrase}'"

        return False, None