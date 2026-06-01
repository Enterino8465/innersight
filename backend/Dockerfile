# Build context is the REPO ROOT (see docker-compose.yml and .github/workflows/ci.yml):
#     docker build -f backend/Dockerfile .
# The repo root itself is the importable `innersight` package, so it is copied
# into the image as innersight/ and installed via its backend/pyproject.toml.
FROM python:3.13-slim
WORKDIR /app

# System deps for building torch / scientific wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# Recreate the package layout inside the image: <app>/innersight/{__init__.py,backend/}
COPY __init__.py ./innersight/__init__.py
COPY backend/ ./innersight/backend/

# Install the package (pyproject lives in innersight/backend with where=["../.."],
# so the `innersight` package resolves to /app/innersight).
RUN pip install --no-cache-dir -e ./innersight/backend

EXPOSE 5001
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5001/api/status')" || exit 1

CMD ["flask", "--app", "innersight.backend.api", "run", "--host", "0.0.0.0", "--port", "5001"]
