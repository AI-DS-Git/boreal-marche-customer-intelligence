# syntax=docker/dockerfile:1
# ---- builder: resolve and install dependencies into an isolated prefix -----------------
# Pin by digest for reproducible builds:  docker buildx imagetools inspect python:3.11-slim-bookworm
# then replace the tag below with python:3.11-slim-bookworm@sha256:<digest>.
FROM python:3.11-slim-bookworm AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements-api.txt .
RUN pip install --prefix=/install -r requirements-api.txt

# ---- runtime: only the installed packages, the app and the model artifacts -------------
FROM python:3.11-slim-bookworm AS runtime
ARG GIT_SHA=dev
ENV GIT_SHA=${GIT_SHA} PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    STRICT_LIBRARY_VERSIONS=true
RUN useradd --system --uid 10001 --create-home app
WORKDIR /srv
COPY --from=builder /install /usr/local
COPY app ./app
COPY models ./models
COPY docs/head_b_rules.json docs/head_b_popularity.json ./docs/
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).status==200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
