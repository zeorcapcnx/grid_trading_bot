# Multi-stage build for production
FROM python:3.12-slim as builder

# Install uv for fast dependency management
RUN pip install uv

WORKDIR /app
COPY pyproject.toml ./
COPY requirements.txt* ./

# Install dependencies
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install -r pyproject.toml

# Production stage
FROM python:3.12-slim as runtime

# Create non-root user
RUN adduser --disabled-password --gecos '' trader && \
    mkdir -p /app/logs /app/data && \
    chown -R trader:trader /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set up environment
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

WORKDIR /app
USER trader

# Copy application code
COPY --chown=trader:trader . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

EXPOSE 8080

# Run the bot
CMD ["python", "main.py", "--config", "config/config_live.json"]