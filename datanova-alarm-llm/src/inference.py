from transformers import pipeline
import os

def load_pipes():
    classifier = pipeline("text-classification", model=os.getenv("CLASSIFIER_MODEL", "distilbert-base-uncased-finetuned-sst-2-english"))
    summarizer = pipeline("summarization", model=os.getenv("SUMMARIZER_MODEL", "t5-small"))
    return classifier, summarizer

def predict_text(msg: str):
    clf, smz = load_pipes()
    cls = clf(msg)[0]
    sm = smz(msg, max_length=32, min_length=8, do_sample=False)[0]["summary_text"]
    return cls, sm

if __name__ == "__main__":
    c, s = predict_text("ALM-1256: Heater current drift beyond 3C during 20min run; recipe HTR_200.")
    print(c)
    print(s)