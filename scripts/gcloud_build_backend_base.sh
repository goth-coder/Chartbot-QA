#!/usr/bin/env bash
# Builds + pushes the chartqa-backend-base image (backend/Dockerfile.base) to Artifact
# Registry. This is the EXPENSIVE half of the backend image: torch/transformers/spaCy deps
# + the ~1.4 GB of baked ML models (CLIP, deberta-injection, all-MiniLM topic-check). The
# thin backend image (backend/Dockerfile) is FROM this, so a normal deploy rebuilds in
# minutes. Mirrors scripts/gcloud_build_vlm_base.sh.
#
# RUN THIS RARELY — only when one of these changes:
#   - the python base image tag,
#   - a pip dependency (backend/requirements.txt / requirements-guard.txt),
#   - the set of ML models under backend/models/.
# A normal code/config deploy does NOT need this — it just reruns
# scripts/gcloud_deploy_app.sh, which builds the thin `FROM backend-base` image.
#
# PREREQUISITE: backend/models/ must be populated on this machine first (it is gitignored —
# HF serves these models only via its Xet CDN, which fails on Cloud Build, so they're
# downloaded on a dev machine and COPY'd into the base from the build context). See
# backend/models/README.md for the one-time download commands.
#
# Usage:
#   ./scripts/gcloud_build_backend_base.sh [--project PROJECT] [--region REGION]
#
# Prerequisites: gcloud CLI installed and authenticated (`gcloud auth login`).
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(dirname "$0")/_gcloud_common.sh"

PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${GCP_REGION:-us-central1}"
REPO="chartqa"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$PROJECT" ]]; then
  echo "No GCP project set. Pass --project <id> or: gcloud config set project <id>" >&2
  exit 1
fi

# The models are gitignored + not downloadable in Cloud Build — they must be on disk here
# so the repo-root build context includes them. Fail fast with a clear pointer if not.
for m in clip-vit-base-patch32 deberta-v3-base-prompt-injection-v2 all-MiniLM-L6-v2; do
  if [[ ! -f "backend/models/${m}/config.json" ]]; then
    echo "[build-backend-base] ERROR: backend/models/${m} is missing." >&2
    echo "[build-backend-base]   Populate backend/models/ first (one-time, on a machine with" >&2
    echo "[build-backend-base]   normal network access). See backend/models/README.md." >&2
    exit 1
  fi
done

IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/backend-base:latest"
echo "[build-backend-base] project=$PROJECT region=$REGION image=$IMAGE"

echo "[build-backend-base] enabling required APIs..."
gcloud services enable artifactregistry.googleapis.com cloudbuild.googleapis.com \
  --project "$PROJECT" --quiet

if ! gcloud artifacts repositories describe "$REPO" --location "$REGION" \
    --project "$PROJECT" >/dev/null 2>&1; then
  echo "[build-backend-base] creating Artifact Registry repo $REPO..."
  gcloud artifacts repositories create "$REPO" --repository-format=docker \
    --location "$REGION" --project "$PROJECT" \
    --description "Chart-Visual-QA images"
fi

echo "[build-backend-base] building + pushing base image via Cloud Build (uploads ~1.4 GB"
echo "  of models + installs torch/transformers/spaCy — a few minutes; run rarely)..."
gcloud builds submit . \
  --project "$PROJECT" \
  --config backend/cloudbuild.base.yaml \
  --substitutions="_IMAGE=${IMAGE}"

echo "[build-backend-base] pushed: ${IMAGE}"

# A leftover old digest from a previous base build is ongoing storage cost. Clean up the
# untagged ones now that the new :latest is pushed.
cleanup_old_images "${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/backend-base"

echo "[build-backend-base] next: ./scripts/gcloud_deploy_app.sh --project ${PROJECT} --region ${REGION} ..."
echo "[build-backend-base]   (the thin app deploy now builds against this base in minutes.)"
