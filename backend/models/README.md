# backend/models — locally-downloaded HF models (gitignored)

These directories hold the three HuggingFace models the backend loads from disk instead
of downloading at runtime:

| Dir | Used by | Config env | Size |
| --- | --- | --- | --- |
| `clip-vit-base-patch32` | chart gate (`chart_check.py`) | `CHART_CLIP_MODEL` | ~580 MB |
| `deberta-v3-base-prompt-injection-v2` | Layer-2 injection (`guard.py`) | `GUARD_INJECTION_MODEL` | ~715 MB |
| `all-MiniLM-L6-v2` | topic check (`topic_check.py`) | `TOPIC_CHECK_MODEL` | ~88 MB |

## Why they live here (not in git, not downloaded in Cloud Build)

HuggingFace now serves these models' weights **only via its Xet CDN**
(`xet-bridge` / `cdn.hf.co`), whose endpoints fail on Cloud Build's network (403
SignatureError / indefinite stall) — so they **cannot be downloaded during the Cloud
Build image build** at all. They also shouldn't go in Git (LFS quota + clone bloat +
image modularity). So:

- This dir is **gitignored** (`.gitignore`: `backend/models/`).
- The models are **baked into the `chartqa-backend-base` image** (`backend/Dockerfile.base`
  `COPY`s them from the build context) — i.e. they live in Artifact Registry, not Git.
- The thin `backend/Dockerfile` inherits them `FROM backend-base`; `env_config.resolve_model_path`
  turns the relative config paths (`models/...`) into `/app/models/...` at runtime.

## How to (re)populate this dir

Do this **once on a machine with normal network access** (Xet works fine off Cloud
Build) before running `scripts/gcloud_build_backend_base.sh`. From `backend/` with the
venv active:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
# CLIP — pytorch_model.bin + tokenizer/processor configs
snapshot_download("openai/clip-vit-base-patch32", local_dir="models/clip-vit-base-patch32",
    allow_patterns=["*.json","*.txt","pytorch_model.bin","vocab.json","merges.txt",
                    "tokenizer*","preprocessor*","special_tokens*"])
# deberta prompt-injection — safetensors + tokenizer
snapshot_download("protectai/deberta-v3-base-prompt-injection-v2",
    local_dir="models/deberta-v3-base-prompt-injection-v2",
    allow_patterns=["*.json","*.txt","model.safetensors","tokenizer*","spm.model",
                    "special_tokens*","added_tokens*"])
# all-MiniLM topic-check — safetensors + tokenizer
snapshot_download("sentence-transformers/all-MiniLM-L6-v2", local_dir="models/all-MiniLM-L6-v2",
    allow_patterns=["*.json","model.safetensors","tokenizer*","vocab.txt",
                    "sentence_bert_config*","modules.json","special_tokens*"])
PY
```

Then trim non-essential extras to keep the base image small (optional): remove any
`.cache/`, `openvino/`, `onnx/` subdirs and redundant weight formats (e.g. a `pytorch_model.bin`
when a `model.safetensors` is already present).

Verify they load offline before building the base:
```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -c "
from transformers import CLIPModel, AutoModel, pipeline
CLIPModel.from_pretrained('models/clip-vit-base-patch32')
pipeline('text-classification', model='models/deberta-v3-base-prompt-injection-v2', truncation=True)
AutoModel.from_pretrained('models/all-MiniLM-L6-v2')
print('all three load offline OK')"
```
