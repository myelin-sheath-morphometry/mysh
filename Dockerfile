FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install CPU-only PyTorch first (avoids pulling ~2GB CUDA variant)
RUN uv pip install --system --no-cache \
    torch --index-url https://download.pytorch.org/whl/cpu

# Copy and install the mysh package
COPY pyproject.toml README.md ./
COPY mysh/ mysh/
# Use headless OpenCV to avoid GL system dependencies in Docker
RUN uv pip install --system --no-cache ".[webserver]" && \
    uv pip install --system --no-cache opencv-python-headless

# Copy webserver code
COPY mysh_webserver/ mysh_webserver/
COPY models/ models/

# Create working directories
RUN mkdir -p /app/mysh_webserver/uploads /app/mysh_webserver/jobs

# mysh_webserver uses relative imports (from backend_celery import ...)
WORKDIR /app/mysh_webserver

# Allow mysh and mysh_webserver package imports
ENV PYTHONPATH=/app
