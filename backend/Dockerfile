# Backend: Flask API + CLIP chart detector + Layer-2 encoders + Tesseract OCR.
# Served by gunicorn (production mode) via docker-compose.yml / Cloud Run.
FROM python:3.12-slim

# Tesseract OCR engine — the system binary pytesseract shells out to (so the
# "has data values" gate runs without a host install). On PATH here, so the
# backend service sets TESSERACT_CMD="" (see docker-compose.yml).
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU build of torch — the backend runs CLIP on CPU; the GPU is for the vLLM guard.
# Installed first so transformers/detoxify don't pull the much larger CUDA wheel.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Layer-2 guard models pull in transformers/presidio/detoxify and (via `-r
# requirements.txt`) the app + CLIP deps. Cached as its own layer.
#
# `pip uninstall -y hf_xet` is the DETERMINISTIC fix for the CLIP/model-bake stalls that
# plagued this project's builds (2026-07). Root cause: recent huggingface_hub versions
# declare hf-xet as a platform-conditional dependency (auto-installed on x86_64), which
# makes model downloads use the Xet transfer protocol (xet-bridge / cdn.hf.co) instead of
# plain HTTPS. Those Xet endpoints fail/hang on Cloud Build's network (a 403 SignatureError
# or an indefinite stall — confirmed in build logs), while the ordinary HF Hub HTTPS
# endpoints work fine. The intended off-switch, HF_HUB_DISABLE_XET=1, did NOT take effect
# in the build for reasons we couldn't pin down, so we remove the package outright:
# huggingface_hub's is_xet_available() is `HF_HUB_DISABLE_XET off AND is_package_available("hf_xet")`,
# so with the package gone, downloads ALWAYS use the classic HTTP path — no flag to honor,
# no version-behavior to depend on. (Xet works fine on normal networks; this is purely a
# Cloud Build network incompatibility.)
COPY requirements.txt requirements-guard.txt ./
RUN pip install --no-cache-dir -r requirements-guard.txt \
    && pip uninstall -y hf_xet hf-xet 2>/dev/null || true \
    && python -m spacy download en_core_web_lg

# Run as a non-root user (security hardening, §3.7). Create it BEFORE the model-precache
# steps and point every ML cache at a dir it owns, then switch to it — so the weights are
# cached (and found at runtime) under the runtime user, not root's unreadable ~/.cache.
# HF_HOME covers transformers/CLIP/deberta; TORCH_HOME covers detoxify's torch.hub download.
#
# HF_HUB_DISABLE_XET=1 is the REAL fix for the CLIP-download stalls that plagued this
# project's builds (2026-07): a transitive dep bump pulled in `hf-xet`, which makes
# huggingface_hub use the Xet transfer protocol (cas-*.xethub.hf.co) by default instead of
# plain HTTPS. Those Xet endpoints stall indefinitely on Cloud Build's network (the
# ordinary HF Hub HTTPS endpoints do NOT — confirmed: a CLIP config fetch returns in
# ~0.2s, and the token is valid, so it was never rate-limiting). Forcing the classic HTTP
# download path, which always worked before hf-xet existed, unblocks the model bakes.
ENV HF_HOME=/opt/ml-cache/hf \
    TORCH_HOME=/opt/ml-cache/torch \
    XDG_CACHE_HOME=/opt/ml-cache/xdg \
    HF_HUB_DISABLE_XET=1
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /opt/ml-cache && chown -R appuser:appuser /opt/ml-cache /app
USER appuser

# Optional HF token for the model-precache RUN steps below (build-time only, injected via
# Cloud Build's Secret Manager integration — see backend/cloudbuild.yaml; NEVER via
# --substitutions/--build-arg on the command line, which Cloud Build logs in plaintext).
# Anonymous HF Hub requests are rate-limited and have repeatedly stalled builds in this
# project past the 60min Cloud Build timeout; a token avoids both. NOT baked into the
# final image as a readable env var — this ENV is overridden to empty below, in the same
# final layer sequence, before COPY; `docker history`/an image inspection never shows a
# token value. Empty ARG (the default, e.g. a local `docker build` with no --build-arg)
# = anonymous requests, unchanged behavior.
ARG HF_TOKEN=
ENV HF_TOKEN=${HF_TOKEN}

# Pre-cache CLIP weights at build so the first request isn't a ~600 MB download.
RUN python -c "from transformers import CLIPModel, CLIPProcessor; \
    CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
    CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"

# Pre-cache the Layer-2 guard encoders at build (toxicity = detoxify/toxic-bert,
# prompt-injection = deberta) so the backend doesn't download them at boot — warmup
# becomes instant instead of a multi-minute runtime download. Presidio (PII) uses the
# spaCy model already installed above. Keep these model ids in sync with .env defaults
# (GUARD_TOXICITY_MODEL / GUARD_INJECTION_MODEL); a non-default value re-downloads once.
RUN python -c "from detoxify import Detoxify; Detoxify('original')" \
    && python -c "from transformers import pipeline; \
       pipeline('text-classification', \
                model='protectai/deberta-v3-base-prompt-injection-v2', truncation=True)"

# NOTE: the topic-check embedding model (all-MiniLM-L6-v2) is NOT downloaded here, unlike
# the models above. Its weights are served only via HF's Xet CDN, which fails on Cloud
# Build's network (403 SignatureError), so it can't be baked from the Hub. It's committed
# to the repo (backend/models/all-MiniLM-L6-v2, Git LFS) and arrives via the COPY below;
# topic_check.py loads it from that local path (TOPIC_CHECK_MODEL=models/all-MiniLM-L6-v2).

# Clear the token now that every model-precache step is done — must not persist as a
# readable env var in the final image. Nothing after this point needs HF access anyway
# (HF_HUB_OFFLINE=1 below forbids it).
ENV HF_TOKEN=

COPY --chown=appuser:appuser . .

# All guard/CLIP weights are baked into the image above — never phone home to HF Hub at
# runtime. Without this, every model init still did live HEAD/GET cache-validation calls
# to huggingface.co (confirmed via Cloud Run logs, 2026-07-10: ~10-70s of network round
# trips, sometimes 429-throttled, on the very first /api/ask after a cold start) even
# though the weights were already on disk — same class of bug already fixed for
# vlm_service/Dockerfile, just never applied here.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 5000
# gunicorn, not Flask's dev server: ratelimit.py/answer_cache.py/redis_client.py were
# already designed around "shared across gunicorn workers" (their own comments say so),
# but the CMD never actually used it — `python app.py` runs Werkzeug's single-threaded
# dev server, so Cloud Run's default --concurrency=80 was a lie the container couldn't
# back up (confirmed: a concurrent /api/ask landed on a cold-starting process and 503'd
# while an already-warm instance sat idle). One worker + threads (not multiple worker
# processes) so the ~1-2GB of guard/CLIP models (warmed via gunicorn.conf.py's
# post_worker_init hook) aren't duplicated in memory per worker. Shell form so
# ${PORT:-...} expands (Cloud Run injects PORT; local/docker-compose falls back to 5000).
CMD gunicorn -c gunicorn.conf.py -b 0.0.0.0:${PORT:-5000} app:app
