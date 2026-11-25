"""FastAPI app: retrieval + xG with traceable evidence.

    uv run uvicorn app.main:app

The pipeline (encoder, index, xG scores) is loaded once at startup (lifespan).
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):          # no package install: put src/ (and scripts/ for the ANN class) on the path
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fastapi import FastAPI, HTTPException                      # noqa: E402
from fastapi.responses import FileResponse                      # noqa: E402

from app.schemas import AnswerRequest, AnswerResponse, Health, SearchRequest, SearchResponse, ShotDetail   # noqa: E402
from football_rag.cnncompare import load_cnn_comparison, shot_tensor   # noqa: E402
from football_rag.evaldata import load_evaluation               # noqa: E402
from football_rag.generate import Generator, build_example      # noqa: E402
from football_rag.pipeline import Pipeline                      # noqa: E402

STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = Pipeline(device="cpu")
    app.state.generator = Generator()          # the language model is loaded on the first /answer call
    yield


app = FastAPI(title="Football shot retrieval + xG", version="0.1.0", lifespan=lifespan,
              description="Natural-language shot search (5 attributes) with xG, actual outcome and source pointers.")


@app.get("/health", response_model=Health)
def health():
    p: Pipeline = app.state.pipeline
    return Health(status="ok", n_documents=len(p.corpus), n_shots=len(p.tab), xg_model="ANN (32,) 5-seed ensemble",
                  encoder="models/minilm_aug", low_confidence_threshold=p.threshold, scope_rule=p.rule.get("chosen", p.rule["name"]))


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    return app.state.pipeline.search(req.query, req.k, req.test_only)


NOTICE = ("Generated text. The model's answer is shown only if it passes an automatic checker (xG values, outcomes, names) and "
          "group-claim rules; otherwise a rule-based answer is shown. Both are heuristic checks, not a guarantee. Check the cited shots.")


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest):
    gen: Generator = app.state.generator
    if not gen.available:
        raise HTTPException(status_code=503, detail="generator checkpoint models/flan_t5_armC not found")
    res = app.state.pipeline.search(req.query, 3, req.test_only)
    ex = build_example(req.query, res)
    out = gen.answer(ex)
    shown = res["evidence"][:len(ex["evidence"])]
    return {"query": req.query, **out, "citations": {str(i): e["shot_id"] for i, e in enumerate(ex["evidence"], 1)},
            "evidence": shown, "summary": res["summary"], "notice": NOTICE}


@app.get("/evaluation")
def evaluation():
    """Saved evaluation results (xG, retrieval, scope, generation, audit) for the page's Evaluation tab; read from results/."""
    return load_evaluation()


@app.get("/evaluation/cnn")
def evaluation_cnn():
    """ANN vs CNN comparison from the saved Phase 2 predictions: summary table, calibration bins, architecture search, sample shots."""
    return load_cnn_comparison()


@app.get("/spatial/{shot_id}")
def spatial(shot_id: str):
    """The 6x40x60 spatial tensor of a shot and the logistic/ANN/CNN probabilities (saved predictions exist for val/test shots only)."""
    out = shot_tensor(shot_id)
    if out is None:
        raise HTTPException(status_code=404, detail=f"shot_id {shot_id!r} not found")
    return out


@app.get("/shot/{shot_id}", response_model=ShotDetail)
def shot(shot_id: str):
    detail = app.state.pipeline.get_shot(shot_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"shot_id {shot_id!r} not found")
    return detail


@app.get("/", include_in_schema=False)
def index_page():
    return FileResponse(STATIC / "index.html")
