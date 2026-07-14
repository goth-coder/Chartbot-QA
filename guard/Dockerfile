# Llama Guard on CPU via Ollama, with the model BAKED INTO the image so a cold
# start never re-downloads it (key for scale-to-zero). Build: docker compose build guard.
#
# NOTE: do NOT mount a volume over /root/.ollama for this image — an empty volume
# would shadow the baked-in model and force a re-pull. (See docker-compose.yml.)
FROM ollama/ollama:latest

# Pre-pull the guard model at build time: start the server, wait until it's up,
# pull, then let the build step end (the background server stops with the shell;
# the downloaded weights persist in the image layer under /root/.ollama).
RUN ollama serve & \
    until ollama list >/dev/null 2>&1; do sleep 1; done; \
    ollama pull llama-guard3:1b

# This service runs CPU-only (no --gpu on Cloud Run, no GPU passthrough in
# docker-compose) — but Ollama's default behavior probes for CUDA and Vulkan GPUs on
# every cold start regardless, each attempt gated by a ~30s watchdog before falling
# through to the next backend. Confirmed via real Cloud Run logs (2026-07-10): CUDA v12
# probe timeout -> CUDA v13 probe timeout -> Vulkan probe timeout -> falls back to CPU,
# ~90s total before the server is even ready to load the model. Since the backend's
# GUARD_LLM_TIMEOUT (20s) is far shorter than that, Layer 3 timed out and failed open on
# EVERY request in production — Llama Guard was never actually screening anything.
# OLLAMA_LLM_LIBRARY=cpu skips GPU auto-discovery entirely and goes straight to the CPU
# backend, cutting cold start by ~90s and making Layer 3 actually reachable within
# GUARD_LLM_TIMEOUT.
ENV OLLAMA_LLM_LIBRARY=cpu

# Keep the model resident for the life of the instance. Ollama's default unloads a model
# after 5 minutes idle, so on a bursty demo the SECOND request after a lull pays a full
# ~15-18s re-load even when the container itself is still warm. OLLAMA_KEEP_ALIVE=-1
# disables that idle-unload; the 1B model stays in RAM until the instance scales to zero,
# so once warmed (at boot / via the /api/guard/warm ping) every subsequent request within
# the instance's life hits a warm model (~0.5-1.2s). Costs nothing extra — the instance is
# already billed while alive; this just avoids re-loading into memory it already holds.
ENV OLLAMA_KEEP_ALIVE=-1

# Override the base image's ENTRYPOINT/CMD (["/bin/ollama"] ["serve"], hardcoded to
# :11434) with a shell wrapper honoring $PORT — same reasoning as vlm_service/Dockerfile:
# Cloud Run injects PORT (usually 8080) and requires the container to listen on exactly
# that value; RunPod/local docker-compose has no $PORT, so this falls back to Ollama's
# normal 11434 (matches GUARD_LLM_URL=http://guard:11434 everywhere else unchanged).
ENTRYPOINT []
CMD ["/bin/sh", "-c", "OLLAMA_HOST=0.0.0.0:${PORT:-11434} exec ollama serve"]
