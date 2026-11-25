#!/usr/bin/env python
"""Phase 4, step 3: embed all 10,233 documents with models/minilm_aug, save the index, sanity-check on VAL.

    uv run python scripts/phase4_build_index.py           # CPU

Writes models/index/{embeddings.npy, ids.json, index_meta.json} and results/phase4/retrieval_check.json.

Checks (validation queries only; test queries are never used):
  * nP@10 on val template queries (text_pairs.jsonl, split == val), searched in the FULL index and, as in
    Phase 3, inside the val-only corpus. nP@10 = hits in top 10 / min(10, n_relevant).
  * LOW-CONFIDENCE THRESHOLD (rule fixed before looking): the 5th percentile of the top-1 cosine
    similarity over the val template queries (full index). A query whose best match scores below that is
    flagged. Plausibility is then checked on para_dev (hand-written paraphrases, even indices of PARA in
    notebook 03) and a few out-of-scope sentences.
"""
import ast
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from football_rag import config as C                                   # noqa: E402
from football_rag import retrieval as R                                # noqa: E402

OUT_OF_SCOPE = ["what is the weather in Paris tomorrow", "who won the world cup in 2010",
                "recipe for chocolate cake", "free kicks taken in minute 85", "all shots by Lionel Messi"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def para_dev_from_notebook():
    """Read the PARA literal from notebook 03 (no copy) and return the dev half (even indices)."""
    nb = json.load(open(C.ROOT / "notebooks" / "03_biencoder_clean.ipynb", encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if src.lstrip().startswith("PARA = ["):
            for node in ast.parse(src).body:
                if isinstance(node, ast.Assign) and node.targets[0].id == "PARA":
                    return ast.literal_eval(node.value)[0::2]
    raise RuntimeError("PARA not found in notebook 03")


def np10(top_idx, rel_mask_fn, n_rel):
    hits = sum(rel_mask_fn(i) for i in top_idx[:10])
    return hits / min(10, n_rel)


def main():
    t0 = time.time()
    corpus = R.load_corpus()
    assert len(corpus) == 10233
    enc = R.load_encoder("cpu")
    emb = R.embed(enc, [r["document"] for r in corpus])
    C.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(C.INDEX_DIR / "embeddings.npy", emb)
    (C.INDEX_DIR / "ids.json").write_text(json.dumps([{"shot_id": r["shot_id"], "row_idx": r["row_idx"]} for r in corpus]),
                                          encoding="utf-8")
    (C.INDEX_DIR / "index_meta.json").write_text(json.dumps({
        "n_docs": len(corpus), "dim": int(emb.shape[1]), "encoder": "models/minilm_aug", "max_seq_length": C.MAX_SEQ_LENGTH,
        "normalize_embeddings": True, "device": "cpu", "corpus_sha256": sha(C.CORPUS),
        "encoder_sha256": sha(C.ENCODER_DIR / "model.safetensors")}, indent=2), encoding="utf-8")
    print(f"embedded {emb.shape} in {time.time() - t0:.0f}s")

    index = R.load_index()
    attrs = [r["doc_attrs"] for r in corpus]
    splits = np.array([r["split"] for r in corpus])
    val_ix = np.where(splits == "val")[0]

    # ---- val template queries
    pairs = [json.loads(l) for l in open(C.PAIRS, encoding="utf-8")]
    vq = {}
    for p in pairs:
        if p["split"] == "val":
            vq.setdefault(p["query"], p["attrs"])
    qtexts = list(vq)
    qemb = R.embed(enc, qtexts)
    sims = index.similarities(qemb)                                      # (Q, N)
    full_scores, val_scores, top1, n_rel_full = [], [], [], []
    for qi, q in enumerate(qtexts):
        a = vq[q]
        rel = np.fromiter((R.is_relevant(a, d) for d in attrs), bool, len(attrs))
        # full index
        top = np.argsort(-sims[qi])[:10]
        full_scores.append(rel[top].sum() / min(10, rel.sum()))
        top1.append(sims[qi].max())
        # val-only corpus (Phase 3 protocol)
        sv = sims[qi][val_ix]
        topv = val_ix[np.argsort(-sv)[:10]]
        val_scores.append(rel[topv].sum() / min(10, rel[val_ix].sum())) if rel[val_ix].sum() else None
        n_rel_full.append(int(rel.sum()))
    top1 = np.array(top1)
    thr = float(np.percentile(top1, 5))
    res = {"n_unique_val_template_queries": len(qtexts), "nP@10_full_index": float(np.mean(full_scores)),
           "nP@10_val_corpus": float(np.mean([s for s in val_scores if s is not None])),
           "n_queries_below_1.0_full_index": int(np.sum(np.array(full_scores) < 1.0)),
           "median_n_relevant_full_index": float(np.median(n_rel_full)),
           "top1_sim_val_template": {"min": float(top1.min()), "p5": thr, "median": float(np.median(top1)), "max": float(top1.max())},
           "low_confidence_threshold": thr}
    print(json.dumps(res, indent=2))

    # ---- plausibility of the threshold on para_dev and out-of-scope sentences (info only)
    pd_items = para_dev_from_notebook()
    pe = R.embed(enc, [q for q, _ in pd_items])
    ps = index.similarities(pe)
    rows = []
    for (q, a), s in zip(pd_items, ps):
        rel = np.fromiter((R.is_relevant(a, d) for d in attrs), bool, len(attrs))
        if rel.sum() == 0:
            continue
        top = np.argsort(-s)[:10]
        rows.append({"query": q, "top1_sim": float(s.max()), "nP@10": float(rel[top].sum() / min(10, rel.sum()))})
    oe = index.similarities(R.embed(enc, OUT_OF_SCOPE)).max(1)
    res["para_dev"] = {"n": len(rows), "nP@10": float(np.mean([r["nP@10"] for r in rows])),
                       "share_below_threshold": float(np.mean([r["top1_sim"] < thr for r in rows])),
                       "mean_top1_sim_if_nP10_eq_1": float(np.mean([r["top1_sim"] for r in rows if r["nP@10"] == 1.0])),
                       "mean_top1_sim_if_nP10_lt_1": float(np.mean([r["top1_sim"] for r in rows if r["nP@10"] < 1.0])) if any(r["nP@10"] < 1 for r in rows) else None,
                       "rows": rows}
    res["out_of_scope_top1_sim"] = dict(zip(OUT_OF_SCOPE, map(float, oe)))
    print({k: v for k, v in res["para_dev"].items() if k != "rows"})
    print("out of scope top-1 sims:", {k: round(v, 3) for k, v in res["out_of_scope_top1_sim"].items()}, "| threshold", round(thr, 3))
    C.RESULTS_P4.mkdir(parents=True, exist_ok=True)
    (C.RESULTS_P4 / "retrieval_check.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
