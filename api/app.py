from fastapi import FastAPI, HTTPException
from .schemas import PromptInput, ScanResult
from src.monitors import SecurePromptPipeline

app = FastAPI(title="SecurePrompt API", version="0.2.0")

# Initialize the pipeline ONCE when the app starts
# This prevents reloading the heavy BERT/GPT models on every request
print("Loading SecurePrompt Pipeline... Please wait.")
pipeline = SecurePromptPipeline()
print("SecurePrompt Ready!")


@app.get("/")
def home():
    return {"message": "SecurePrompt API is running. Send POST requests to /scan."}


@app.post("/scan", response_model=ScanResult)
def scan_prompt(input_data: PromptInput):
    """
    Scans a user prompt for injection attacks, leakage, and policy violations.
    """
    try:
        # 1. Run the logic from src/monitors/integration.py
        result = pipeline.scan_input(input_data.prompt)

        # 2. Return the dictionary exactly as schemas.py expects
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))