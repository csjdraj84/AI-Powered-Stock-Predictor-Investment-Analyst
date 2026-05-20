# ── Stage 1: builder ─────────────────────────────────────────────────────────
# Build on a full image so heavy compilation (tensorflow, etc.) succeeds,
# then copy only the installed packages into the slim runtime image.
FROM --platform=linux/amd64 python:3.11-slim AS builder

WORKDIR /build

# System deps needed only during compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps into an isolated prefix so we can copy them cleanly
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM --platform=linux/amd64 python:3.11-slim AS runtime

# Metadata labels (visible in Hostinger Docker Manager UI)
LABEL maintainer="dhanarajjayapalan" \
      org.opencontainers.image.title="AI Stock Prediction & Investment Analyst" \
      org.opencontainers.image.description="Streamlit app for AI-powered stock prediction and investment analysis" \
      org.opencontainers.image.version="1.0.0"

WORKDIR /app

# Runtime system deps only (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Create a non-root user for security (Hostinger best practice)
RUN useradd --create-home --shell /bin/bash appuser

# Copy application source (respects .dockerignore)
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Streamlit port
EXPOSE 8501

# Healthcheck — Hostinger Docker Manager uses this to determine container health
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail --silent http://localhost:8501/_stcore/health || exit 1

# Streamlit configuration via env (can be overridden in Hostinger env vars panel)
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]
