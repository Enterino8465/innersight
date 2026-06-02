# Build context is the REPO ROOT (see docker-compose.yml and .github/workflows/ci.yml):
#     docker build -f backend/Dockerfile .
# backend/ is the package root: it holds pyproject.toml and the importable
# `innersight` package (backend/innersight/). Its contents are copied to /app
# and installed in place.
FROM python:3.13-slim
WORKDIR /app

# System deps for building torch / scientific wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# Copy the backend package root (pyproject.toml + innersight/ package) into /app.
COPY backend/ ./

# Install the package (pyproject lives at /app with where=["."], so the
# `innersight` package resolves to /app/innersight).
RUN pip install --no-cache-dir -e .

EXPOSE 5001
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5001/api/status')" || exit 1

CMD ["flask", "--app", "innersight.api", "run", "--host", "0.0.0.0", "--port", "5001"]
