# Backend THIN image: FROM the prebuilt chartqa-backend-base (heavy deps + baked ML
# models), so a normal code/config deploy rebuilds in ~2-3 min without reinstalling
# torch/transformers or re-fetching ~1.4 GB of models. See backend/Dockerfile.base +
# scripts/gcloud_build_backend_base.sh for the base (rebuilt rarely).
#
# Build context is backend/ (so backend/.dockerignore governs it). BACKEND_BASE_IMAGE is
# passed by scripts/gcloud_deploy_app.sh via backend/cloudbuild.yaml; the default is just
# for a bare local `docker build backend/`.
ARG BACKEND_BASE_IMAGE=chartqa-backend-base:latest
FROM ${BACKEND_BASE_IMAGE}

WORKDIR /app

# The app source only. Heavy deps + models are inherited from the base. backend/.dockerignore
# excludes models/ so this COPY does NOT re-ship the 1.4 GB already in the base (and doesn't
# shadow it), plus .venv/tests/etc. Everything the runtime reads — code, gunicorn.conf.py,
# and the model dirs (from the base) — ends up under /app.
COPY --chown=appuser:appuser . /app/

# Never phone home to HF Hub at runtime — the models are all on local disk (baked into the
# base). Without this, every model init still did live HEAD/GET cache-validation calls to
# huggingface.co on the first request after a cold start (confirmed via Cloud Run logs).
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 5000
# gunicorn (not Flask's dev server): ratelimit/answer_cache/redis_client are designed
# around "shared across gunicorn workers". One worker + threads (not multiple worker
# processes) so the ~1-2 GB of guard/CLIP models (warmed via gunicorn.conf.py's
# post_worker_init hook) aren't duplicated in memory. Shell form so ${PORT:-...} expands
# (Cloud Run injects PORT; local/docker-compose falls back to 5000).
CMD gunicorn -c gunicorn.conf.py -b 0.0.0.0:${PORT:-5000} app:app
