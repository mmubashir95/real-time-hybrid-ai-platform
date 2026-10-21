FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
# Install the CPU-only torch wheel first; the default Linux wheel bundles CUDA
# libraries (several GB) that the CPU-only embedding pipeline never uses.
# "2.14.1+cpu" satisfies the "torch==2.14.1" pin in requirements.txt.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 \
    && pip install --no-cache-dir --requirement requirements.txt

RUN useradd --create-home --uid 10001 appuser

COPY --chown=appuser:appuser app ./app
# Ship the immutable SQL history and its runner so Compose and deployment jobs
# execute the same migration path before starting the API.
COPY --chown=appuser:appuser scripts ./scripts
COPY --chown=appuser:appuser migrations ./migrations
COPY --chown=appuser:appuser schema.sql ./schema.sql

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
