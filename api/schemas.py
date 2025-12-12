from pydantic import BaseModel
from typing import Optional, Dict, Any

class PromptInput(BaseModel):
    prompt: str
    user_id: Optional[str] = "anonymous"

class ScanResult(BaseModel):
    status: str          # "PASS" or "BLOCK"
    reason: Optional[str] = None
    metrics: Dict[str, Any]
    warnings: Optional[str] = None