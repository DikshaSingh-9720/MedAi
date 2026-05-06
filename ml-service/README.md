# ml-service

FastAPI prediction service used by `server` via `ML_SERVICE_URL`.

## Setup

```bash
cd ml-service
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## Model files

- Put your trained model at `models/your_model.keras`
- Put class labels at `models/labels.json`

Example labels file:

```json
{ "classes": ["Normal", "Abnormal"] }
```
