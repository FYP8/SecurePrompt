import base64
import binascii
class EncodingPatternDetector:
    def __init__(self):
        self.min_length = 20
    def scan(self,text):
        if len(text) < self.min_length:
            return False,None,None
        try:
            decoded_bytes = base64.b64decode(text,validate = True)

            decoded_text = decoded_bytes.decode('utf-8')
            return True,decoded_text, "Obfuscation Detected: Input was Base64 Encoded"
        except(binascii.Error, UnicodeDecodeError):
            return False,None,None