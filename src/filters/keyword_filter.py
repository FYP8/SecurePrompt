class KeywordFilter:
    def __init__(self):
        self.blocklist = [
            "ignore previous instructions",
            "ignore all instructions",
            "system prompt",
            "you are not a language model",
            "do anything now",
            "dan mode",
            "jailbreak",
            "unfiltered",
            "never refuse",
            "act as an adversary",
            "admin access",
            "developer mode"
        ]
    def scan(self,text):
        text_lower = text.lower()
        for keyword in self.blocklist:
            if keyword in text_lower:
                return True,keyword
        return False,None