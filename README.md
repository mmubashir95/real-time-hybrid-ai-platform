# real-time-hybrid-ai-platform
Real-Time Hybrid AI Support &amp; Operations Platform

## Run locally

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your-key
uvicorn app.main:app --reload
```

The app reads `OPENAI_API_KEY` from the environment. A `.env` file is not loaded automatically.
