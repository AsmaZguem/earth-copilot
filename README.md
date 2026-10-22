# Earth Copilot

Ask a question about a place in plain language — *"How did vegetation change around this mine between
2020 and 2025?"* — and get a satellite analysis on a map, with the choice of spectral index explained
and cited from remote-sensing documentation.

Runs fully locally on free and open-source tools: a local LLM through Ollama, free Sentinel-2 imagery
from Element84 Earth Search, and Qdrant for retrieval.

## How it works

```
Question → Agent (LangGraph + local LLM)
   ├─ geocode(place)            → Nominatim (OpenStreetMap)
   ├─ search_docs(question)     → RAG over spectral-index catalogue + Sentinel-2 / Landsat docs
   ├─ search_imagery(area, dates, clouds) → Earth Search STAC
   ├─ compute_index(scene, index)         → odc-stac + xarray
   └─ change_detection(date1, date2)
→ answer with citations + georeferenced PNG → MapLibre map with before/after slider
```

## Status

| Step | Status |
|---|---|
| 0. Setup, LLM tool-calling check | ✅ code ready |
| 1. Imagery → index → PNG pipeline (no LLM) | ✅ code ready |
| 2. RAG over index catalogue and docs, with evaluation | ⏳ |
| 3. Agent | ⏳ |
| 4. Map UI | ⏳ |
| 5. Evaluation (with vs without RAG), demo | ⏳ |

## Setup

Requirements: [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com), Docker.
A 7B model runs on CPU with 16 GB of RAM (slow but usable); no GPU needed.

```bash
uv sync
cp .env.example .env
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
docker compose up -d        # Qdrant (needed from step 2)
```

## Try it

```bash
# Step 1: least cloudy summer scene over Lac Tremblant (Quebec) -> NDVI PNG in outputs/
uv run python scripts/index_demo.py
uv run python scripts/index_demo.py --index MNDWI --bbox -74.62 46.18 -74.54 46.26

# Step 0: does the local model call tools correctly?
uv run python scripts/check_llm.py
```

To use a hosted free tier instead of Ollama (for example to record a faster demo), set
`LLM_PROVIDER=groq` or `gemini` in `.env`, install the extra (`uv sync --extra groq`) and set the API key.

## Data sources

- Sentinel-2 L2A via [Element84 Earth Search](https://earth-search.aws.element84.com/v1) (free, no account)
- Spectral indices: [Awesome Spectral Indices](https://github.com/awesome-spectral-indices/awesome-spectral-indices)
- Geocoding: [Nominatim](https://nominatim.org) (respect the 1 request/second usage policy)
