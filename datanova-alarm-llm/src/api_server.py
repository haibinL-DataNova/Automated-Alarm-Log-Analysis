from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os

from transformers import pipeline

app = FastAPI(title="DataNova Alarm Log LLM API", version="0.1.0")

# Load models (use env vars to switch to your fine-tuned checkpoints)
CLASSIFIER_MODEL = os.getenv("CLASSIFIER_MODEL", "distilbert-base-uncased-finetuned-sst-2-english")
SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "t5-small")

clf = pipeline("text-classification", model=CLASSIFIER_MODEL)
summarizer = pipeline("summarization", model=SUMMARIZER_MODEL)

class AlarmItem(BaseModel):
    message: str
    recipe: Optional[str] = None
    alarm_code: Optional[str] = None
    tool_id: Optional[str] = None

class PredictRequest(BaseModel):
    items: List[AlarmItem]

@app.get("/health")
def health():
    return {"status": "ok"}

def map_to_severity(label: str, score: float) -> str:
    # Very simple mapper: replace with your fine-tuned label set (normal/warning/critical)
    if label.upper().startswith("NEG"):  # NEGATIVE
        return "warning" if score < 0.9 else "critical"
    return "normal"

@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    results = []
    for it in req.items:
        cls = clf(it.message)[0]
        sev = map_to_severity(cls["label"], cls["score"])
        # keep summaries short for API responsiveness
        sm = summarizer(it.message, max_length=32, min_length=8, do_sample=False)[0]["summary_text"]
        results.append({
            "alarm_code": it.alarm_code,
            "tool_id": it.tool_id,
            "recipe": it.recipe,
            "severity": sev,
            "classification_raw": cls,
            "summary": sm,
            "message": it.message,
        })
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api_server:app", host="127.0.0.1", port=8000, reload=True)