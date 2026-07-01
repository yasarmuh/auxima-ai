# NVIDIA dev models (build.nvidia.com) — DEV / SYNTHETIC ONLY

Per **ADR-OD-NVIDIA-dev-llm**: NVIDIA's hosted API is a US-cloud egress path. Use it only to iterate on
prompts and benchmark frontier models during development, against **synthetic / non-personal data**.
Ollama stays the production default; this never touches `bootstrap.py`.

## 1. Env setup (sourceable, git-ignored)

Create `.env.nvidia` (matches `.env.*`, so git ignores it) and fill in your key:

```bash
cat > .env.nvidia <<'EOF'
export AUXIMA_DEV_LLM_ENABLED=1
export NVIDIA_API_KEY=nvapi-REPLACE_ME
export NVIDIA_DEV_MODEL=meta/llama-3.1-8b-instruct
export AUXIMA_SIDECAR_SHARED_SECRET=REPLACE_WITH_SITE_SECRET
EOF
$EDITOR .env.nvidia
source .env.nvidia
```

`AUXIMA_SIDECAR_SHARED_SECRET` is only needed for the live server (step 3); it must equal the Frappe
site's `site_config.json` → `auxima_sidecar_secret`. The NVIDIA vars are read from `os.environ`
directly, so they must be **exported** (a bare `.env` does NOT reach them).

## 2. Offline harness — draft against a model, no server

```bash
./.venv/Scripts/python.exe scripts/dev_nvidia.py --message "Is my motor policy still active?"
./.venv/Scripts/python.exe scripts/dev_nvidia.py --kind email --purpose "follow up on renewal" --language ar
./.venv/Scripts/python.exe scripts/dev_nvidia.py --model deepseek-ai/deepseek-r1 --message "..."
```

## 3. Live dev server — the "Draft with AI" button returns a real NVIDIA draft

```bash
source .env.nvidia
./.venv/Scripts/python.exe scripts/dev_serve_nvidia.py     # binds 0.0.0.0:8088
```

The Frappe bench (already configured with `auxima_sidecar_url=http://host.docker.internal:8088`) will
reach it, so the inbox's **Draft with AI** button drafts via NVIDIA — for **synthetic** conversations
only. Ctrl-C to stop; the normal `uvicorn auxima_ai.main:app` (Ollama-first) is the real path.
