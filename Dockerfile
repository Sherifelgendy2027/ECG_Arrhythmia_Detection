# ==============================================================
# Stage 1 — Base OS
# Rarely changes. Sets platform, env, system libs.
# ==============================================================
# Platform is enforced at build time via: docker build --platform linux/amd64
# Omitting --platform here keeps the Dockerfile portable and warning-free.
FROM python:3.11-slim

# Prevent .pyc files and ensure logs are flushed immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System libraries:
#   libgomp1       — required by LightGBM (OpenMP threading)
#   libglib2.0-0   — required by some CV/image libs
#   libgl1         — required if PIL/OpenCV is used for X-ray preprocessing
# Cleanup apt lists in the same layer to avoid bloating the image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================
# Stage 2 — CPU-Only PyTorch (isolated layer)
# Changes only on a torch/torchvision version bump.
# Isolated so the ~700 MB download is cached across code edits.
# ==============================================================
RUN pip install --no-cache-dir \
        "torch>=2.5.0" \
        "torchvision>=0.20.0" \
        --index-url https://download.pytorch.org/whl/cpu

# ==============================================================
# Stage 3 — All other production dependencies
# Only invalidated when requirements-prod.txt changes.
# Copy only the requirements file first (not the whole source).
# ==============================================================
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# ==============================================================
# Stage 4 — Application source
# Invalidated on every code change, but layers 1-3 are reused.
# ==============================================================
COPY . /app

# Document the port (Render injects $PORT at runtime)
EXPOSE 8000

# JSON array form (satisfies JSONArgsRecommended) with explicit sh -c so
# that ${PORT:-8000} is still evaluated by the shell at container start.
# Render injects $PORT; fallback to 8000 for local `docker run` tests.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
