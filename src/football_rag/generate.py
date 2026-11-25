"""Grounded answer generation for the API: retrieval result -> flan-t5 input -> model answer -> verify / fall back.

`build_example` turns a `Pipeline.search(query, 3, ...)` result into the same example record the Phase 5 data used
(gen_inputs_*.jsonl): the model input string plus the evidence block that the checkers score against.
`Generator` loads the fine-tuned checkpoint (models/flan_t5_armC) lazily and decodes exactly as in the evaluation
(beam 4, max_new_tokens 128, no_repeat_ngram_size 3). `answer()` returns the text that is shown to the user, which is
the model's answer only if it passes the Phase 5 checker and the Phase 6 group rules (`verify.choose_answer`),
otherwise a deterministic template built from the evidence.
"""
import os
import threading
from pathlib import Path

import numpy as np

from football_rag.gen_format import evidence_for_model, fold_text, format_input, outcome_word, retrieval_status
from football_rag.verify import choose_answer

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "models" / "flan_t5_armC"
GEN = dict(num_beams=4, max_new_tokens=128, no_repeat_ngram_size=3, do_sample=False)
MAX_INPUT_TOKENS = 512


def build_example(query: str, result: dict) -> dict:
    """Example record (see scripts/phase5_build_inputs.py) from a pipeline result with k=3."""
    items = evidence_for_model(result)
    status = retrieval_status(result["summary"], not result["evidence"])
    notes = result["summary"]["unsupported"]
    ev = [{"rank": e["rank"], "shot_id": e["shot_id"], "minute": e["minute"], "team": e["team"], "player": e["player"],
           "document": e["document"], "doc_attrs": e["doc_attrs"], "xg": e["xg"], "outcome": outcome_word(e["document"]),
           "goal": e["goal"], "split": e["split"], "document_model": fold_text(e["document"]),
           "player_model": fold_text(e["player"]) if e["player"] else None, "team_model": fold_text(e["team"])} for e in items]
    return {"query": query, "mode": "refuse" if not items else ("partial" if notes else "answer"),
            "input": format_input(query, items, status), "evidence": ev,
            "summary": {"n_results": len(ev), "mean_xg": float(np.mean([e["xg"] for e in ev])) if ev else None,
                        "n_goals": int(sum(e["goal"] for e in ev))},
            "retrieval": {"status": status, "low_confidence": result["summary"]["low_confidence"], "unsupported": notes,
                          "understood": result["summary"]["understood"]}}


class Generator:
    """Lazy, thread-safe wrapper around the fine-tuned flan-t5-base checkpoint."""

    def __init__(self, ckpt: Path = CKPT, device: str | None = None):
        self.ckpt = Path(ckpt)
        self.device = device or os.environ.get("FOOTBALL_GEN_DEVICE", "cpu")
        self._model = self._tok = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return (self.ckpt / "model.safetensors").exists()

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.ckpt)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.ckpt).to(self.device).eval()
            self._torch = torch

    def generate(self, text: str) -> str:
        with self._lock:
            self._load()
            enc = self._tok([text], return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS).to(self.device)
            with self._torch.no_grad():
                out = self._model.generate(**enc, **GEN)
            return self._tok.batch_decode(out, skip_special_tokens=True)[0]

    def answer(self, example: dict) -> dict:
        raw = self.generate(example["input"])
        text, source, reasons = choose_answer(example, raw)
        return {"answer": text, "source": source, "fallback_reasons": reasons, "raw_model_answer": fold_text(raw)}
