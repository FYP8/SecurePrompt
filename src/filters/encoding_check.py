import base64
import binascii


class EncodingPatternDetector:
    def __init__(self):
        self.min_length = 16  # Slightly lowered to catch shorter attacks

    def scan(self, text):
        """
        Checks for Base64 or Hex encoding.
        Returns: (bool is_encoded, str decoded_text, str method_name)
        """
        if not text or len(text) < self.min_length:
            return False, None, None

        cleaned_text = text.strip()

        # --- Check 1: Base64 ---
        # Base64 strings usually end with '=' or have lengths multiple of 4
        if len(cleaned_text) % 4 == 0 or cleaned_text.endswith("="):
            try:
                decoded_bytes = base64.b64decode(cleaned_text, validate=True)
                decoded_text = decoded_bytes.decode('utf-8')

                # Filter: If decoded text is junk/unprintable, it's likely false positive
                if self._is_readable(decoded_text):
                    return True, decoded_text, "Base64"
            except (binascii.Error, UnicodeDecodeError):
                pass

        # --- Check 2: Hexadecimal ---
        # Hex strings are usually 0-9, A-F and even length
        if self._is_hex(cleaned_text):
            try:
                decoded_bytes = bytes.fromhex(cleaned_text)
                decoded_text = decoded_bytes.decode('utf-8')

                if self._is_readable(decoded_text):
                    return True, decoded_text, "Hex"
            except (ValueError, UnicodeDecodeError):
                pass

        return False, None, None

    def _is_readable(self, text):
        """Helper to ensure we didn't just decode random binary garbage"""
        # We expect at least 70% of characters to be printable
        printable_count = sum(1 for c in text if c.isprintable())
        return printable_count / len(text) > 0.7 if len(text) > 0 else False

    def _is_hex(self, s):
        """Fast check if string is potentially hex"""
        return len(s) % 2 == 0 and all(c in '0123456789abcdefABCDEF' for c in s)