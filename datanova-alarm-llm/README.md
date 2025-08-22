# Automated Alarm Log Analysis (LLM-based)

**Brand:** DataNova

Use fine-tuned LLMs to classify and summarize semiconductor tool alarm logs for faster triage and root-cause analysis.

## Tech Stack
- Python, Hugging Face Transformers
- Azure ML (training/deploy), MLflow
- FastAPI (REST API), Uvicorn
- Grafana (dashboard JSON export)

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run a quick demo API
python src/api_server.py
# open http://127.0.0.1:8000/docs
```
## Project Structure
```
.
├── data/                # sample or synthetic logs
├── src/                 # core code (preprocess, train, inference, api)
├── dashboard/           # grafana dashboards (JSON)
├── .github/workflows/   # CI
├── requirements.txt
├── .gitignore
└── README.md
```

## API Example
POST `/predict` with:
```json
{
  "message": "ALM-1256: Heater current drift beyond 3C during 20min run; recipe HTR_200."
}
```
Returns severity + summary.

## Azure ML (Optional)
- Package the model and push to Azure ML registry.
- Use `azure-ai-ml` to create a job and deployment. (See TODOs in `src/train_model.py`).

## License
MIT