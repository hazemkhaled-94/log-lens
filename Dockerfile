# Inference image (CPU) — self-contained log-lens API server.
#
# Installs a CPU-only torch (the CUDA wheel is never pulled, so the image
# stays lean), and bakes the model in at BUILD time from the Hugging Face
# Hub. Defaults to the published modernlogbert-wce checkpoint:
#
#   docker build -t log-lens .                     # uses the default model
#   docker run -p 8000:8000 log-lens
#
# Override --build-arg MODEL_REPO=<hf_repo_id> to bake in a different model.
# For GPU training on your own data, use Dockerfile.train.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    MODEL_DIR=/app/model

RUN pip install poetry poetry-plugin-export

WORKDIR /app

COPY pyproject.toml poetry.lock ./

# Export the locked deps, drop the torch family, then install a CPU-only
# torch from the PyTorch CPU index alongside everything else. This avoids
# ever downloading the CUDA build (and its bundled nvidia-* packages).
RUN poetry export --only main --without-hashes -f requirements.txt -o /tmp/req.txt \
    && grep -ivE '^torch([ =<>!@;]|$)' /tmp/req.txt > /tmp/req-notorch.txt \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r /tmp/req-notorch.txt

# Install the project itself (entry points) without touching dependencies.
COPY src ./src
RUN pip install --no-deps .

# Bake the trained model into the image. MODEL_REPO is a Hugging Face Hub
# repo id; private repos additionally need a HF_TOKEN build secret.
ARG MODEL_REPO="hazemkhaled-94/modernlogbert-wce"
ARG MODEL_REVISION=main
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='${MODEL_REPO}', revision='${MODEL_REVISION}', \
local_dir='${MODEL_DIR}')"

EXPOSE 8000

CMD ["log-lens", "serve"]
