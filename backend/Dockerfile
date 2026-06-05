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

# Bundle the synthetic demo dataset and default to it, so the container works
# with no env vars. A docker-compose volume + INNERSIGHT_DATA_DIR override this
# when real CERT data is available.
COPY data/synthetic_demo/ /app/demo_data/
ENV INNERSIGHT_DATA_DIR=/app/demo_data

# Entrypoint computes demo baselines on first boot (when no checkpoints exist).
COPY backend/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 5001
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5001/health')" || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["flask", "--app", "innersight.api", "run", "--host", "0.0.0.0", "--port", "5001"]
