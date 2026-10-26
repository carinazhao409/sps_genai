# --------------------------------------------------
# 1. Python base image
# --------------------------------------------------

FROM python:3.13-slim-bookworm

# --------------------------------------------------
# 2. Install system dependencies
# --------------------------------------------------

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# 3. Install uv
# --------------------------------------------------

ADD https://astral.sh/uv/install.sh /uv-installer.sh

RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin:$PATH"

# --------------------------------------------------
# 4. Set working directory
# --------------------------------------------------

WORKDIR /code

# --------------------------------------------------
# 5. Copy dependency configuration
# --------------------------------------------------

COPY pyproject.toml uv.lock README.md /code/

COPY ./src /code/src

# --------------------------------------------------
# 6. Install project dependencies
# --------------------------------------------------

RUN uv sync --frozen --no-dev --python 3.13

# --------------------------------------------------
# 7. Download spaCy English language model
# --------------------------------------------------

RUN uv run --no-sync python -m spacy download en_core_web_lg

# --------------------------------------------------
# 8. Copy application code and trained CNN
# --------------------------------------------------

COPY ./app /code/app

# Keep the trained LSTM checkpoint inside the app directory.

# --------------------------------------------------
# 9. Expose FastAPI port
# --------------------------------------------------

EXPOSE 80

# --------------------------------------------------
# 10. Start FastAPI
# --------------------------------------------------

CMD ["uv", "run", "--no-sync", "fastapi", "run", "app/main.py", "--port", "80"]