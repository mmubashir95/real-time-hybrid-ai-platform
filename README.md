# real-time-hybrid-ai-platform
Real-Time Hybrid AI Support &amp; Operations Platform

## Run locally

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your-key
export MODEL_TIMEOUT_SECONDS=30  # optional, defaults to 30
uvicorn app.main:app --reload
```

The app reads these settings from the environment. A `.env` file is not loaded automatically.

## Test

```bash
pip install -r requirements-dev.txt
python -m unittest discover -s tests
```
