# Tactical Intelligence for Football Shot Data

A capstone prototype that lets an analyst ask about football shots in plain language. It returns the matching real shots with an expected-goals (xG) value for each one, and a short written answer that is checked against those shots.

Built on [StatsBomb Open Data](https://github.com/statsbomb/open-data) with 360 freeze frames.
Author: Avishkar Waikar (a3189970).

## What it does

```
question -> scope check -> bi-encoder search (top 3 shots) -> ANN xG per shot
         -> FLAN-T5 writes a cited answer -> checker + group rules
         -> model text, or a rule-based template if a check fails
```




## Requirements

- Python 3.14 (see `.python-version`) and [uv](https://docs.astral.sh/uv/)
- Git (for the open-data download)
- An NVIDIA GPU is recommended for training. The app and tests run on CPU.

`pyproject.toml` pins torch to the CUDA index. Change the `[[tool.uv.index]]` URL if your CUDA version differs.

## Setup

```powershell
uv sync
```

## Data and models are not in the repository

`data/`, `open-data/` and `models/` are listed in `.gitignore` (about 1.2 GB of models). Rebuild them in this order:

| Step | Command | Output |
|---|---|---|
| 1. Download StatsBomb events with 360 data | `git clone https://github.com/statsbomb/open-data open-data` then `uv run python scripts/fetch_events.py` | `open-data/` |
| 2. Prepare shots, splits, tensors, text pairs | run `notebooks/01_explore_data.ipynb` | `data/processed/` |
| 3. Train the xG models | `notebooks/02_models_clean.ipynb` or `uv run python scripts/phase2_train.py` | `results/` |
| 4. Fine-tune the bi-encoder | `notebooks/03_biencoder_clean.ipynb` | `models/minilm_v1`, `models/minilm_aug` |
| 5. Build the xG model, index and scope rule (about 3 min on CPU) | `uv run python scripts/phase4_build_all.py` | `models/xg_ann`, `models/index`, `results/phase4/` |
| 6. Build generator inputs and targets | `scripts/phase5_build_inputs.py`, `scripts/phase5_targets.py` | `data/processed/gen_*` |
| 7. Fine-tune FLAN-T5 (arm C, about 5 min on GPU) | `uv run python scripts/phase5_armC_train.py` | `models/flan_t5_armC` |

The question sets for the Phase 6 tests are in `data/phase6/` with SHA-256 files. Each script writes a `run_info.json` with versions and seeds.

## Run the app

```powershell
uv run uvicorn app.main:app
```

Open http://127.0.0.1:8000. The page has a search tab and an Evaluation tab.

Endpoints: `/search`, `/shot/{id}`, `/answer`, `/evaluation`, `/evaluation/cnn`, `/spatial/{id}`.

The API shows test-split shots by default, because xG for training shots is scored in-sample.

## Tests

```powershell
uv run pytest
```

The tests cover data formats, API behaviour, source-ID resolution, xG consistency, the checker and verifier (including failing examples) and that the evaluation set has not changed. Some tests need the rebuilt `data/` and `models/` folders.

## Reproduce the evaluation

| Question | Command |
|---|---|
| Per-component generation table, intervals, audit sheet | `uv run python scripts/phase5_eval.py` then `scripts/phase5_report.py` |
| 50-question retrieval and scope test | `uv run python scripts/phase6_query_eval.py` |
| Generator on the 50 questions | `uv run python scripts/phase6_generate_50.py` |
| Group-claim analysis | `uv run python scripts/phase6_group_claims.py` |
| Verify-and-fall-back on 60 new questions | `uv run python scripts/phase6_verify_fresh.py` |
| Report figures | `uv run python scripts/phase6_report_figures.py` |

## Project layout

```
app/                 FastAPI service and web page (app/static/index.html)
src/football_rag/    pipeline, retrieval, scope, xG, generation, checker, verifier
scripts/             phase 2 to phase 6 build and evaluation scripts
notebooks/           data exploration, xG models, bi-encoder training
tests/               pytest suite
data/phase6/         frozen test questions with hashes
results/             saved metrics, tables and figures


```



## Data licence

StatsBomb Open Data is used under StatsBomb's terms. Read their licence and attribution rules before sharing results.
