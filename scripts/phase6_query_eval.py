"""Phase 6 step 3: score 50 independently phrased questions (data/phase6/queries_50.csv, AI-drafted, intended
attributes frozen before the run) on the existing retrieval + scope pipeline, test split only. Nothing is tuned.

Per query: the system's scope decision (out = low confidence or no evidence; partial = evidence plus an
unsupported-part note; in = otherwise) against expected_scope, and for in/partial queries with intended attributes
nP@10 / MRR / P@10 over the top 10 test-split shots (relevant = satisfies EVERY intended attribute, Phase 3 definition).
Also writes the 3-evidence generator inputs for step 4.
Outputs: results/phase6/query_eval.csv, query_eval_summary.json, queries_50_inputs.jsonl, query_eval_run_info.json.
"""
import csv, hashlib, json, platform, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from football_rag.retrieval import is_relevant  # noqa: E402
from phase5_build_inputs import Builder  # noqa: E402

Q = ROOT / "data/phase6/queries_50.csv"
OUT = ROOT / "results/phase6"


def parse_attrs(s):
    return {k: v for k, v in (kv.split("=", 1) for kv in s.split(";") if kv)} if s else None


def main():
    assert hashlib.sha256(Q.read_bytes()).hexdigest() == (ROOT / "data/phase6/queries_50.sha256").read_text().split()[0], "query file changed since freezing"
    rows = list(csv.DictReader(open(Q, encoding="utf-8")))
    b = Builder(); p = b.p
    test_idx = np.where(p.split_arr == "test")[0]
    out, inputs = [], []
    for r in rows:
        attrs = parse_attrs(r["intended_attrs"])
        res = p.search(r["question"], 10, test_only=True)
        s = res["summary"]
        scope = "out" if (s["low_confidence"] or not res["evidence"]) else ("partial" if s["unsupported"] else "in")
        d = dict(id=r["id"], question=r["question"], category=r["category"], expected_scope=r["expected_scope"], system_scope=scope,
                 scope_correct=int(scope == r["expected_scope"]), intended=r["intended_attrs"], n_relevant_test="", p10="", np10="", mrr="")
        if attrs and res["evidence"]:
            rel = [is_relevant(attrs, e["doc_attrs"]) for e in res["evidence"]]
            n_rel = int(sum(is_relevant(attrs, p.corpus[i]["doc_attrs"]) for i in test_idx))
            d["n_relevant_test"] = n_rel
            if n_rel:                                   # a query with no relevant test shot cannot be scored (reported separately)
                d.update(p10=sum(rel) / 10, np10=sum(rel) / min(10, n_rel), mrr=(1 / (rel.index(True) + 1)) if any(rel) else 0.0)
        out.append(d)
        ex = b.example(r["id"], "test", r["category"], r["question"], attrs)
        inputs.append(ex)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "query_eval.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    (OUT / "queries_50_inputs.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in inputs), encoding="utf-8")

    def mean(v):
        v = [x for x in v if x != ""]
        return (float(np.mean(v)), len(v)) if v else (None, 0)
    summ = {"n": len(out), "scope_accuracy": float(np.mean([d["scope_correct"] for d in out])), "by_category": {}, "confusion": {}}
    conf = defaultdict(int)
    for d in out:
        conf[f"expected_{d['expected_scope']}->system_{d['system_scope']}"] += 1
    summ["confusion"] = dict(conf)
    for cat in sorted({d["category"] for d in out}):
        sub = [d for d in out if d["category"] == cat]
        summ["by_category"][cat] = dict(n=len(sub), scope_correct=sum(d["scope_correct"] for d in sub),
                                        np10=mean([d["np10"] for d in sub]), mrr=mean([d["mrr"] for d in sub]), p10=mean([d["p10"] for d in sub]))
    ret = [d for d in out if d["mrr"] != ""]
    summ["unscoreable_no_relevant_test_shot"] = [d["id"] for d in out if d["n_relevant_test"] == 0]
    summ["retrieval_in_or_partial"] = dict(n=len(ret), np10=mean([d["np10"] for d in ret]), mrr=mean([d["mrr"] for d in ret]),
                                           p10=mean([d["p10"] for d in ret]), queries_with_zero_relevant_in_top10=sum(d["mrr"] == 0.0 for d in ret))
    (OUT / "query_eval_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    import transformers
    (OUT / "query_eval_run_info.json").write_text(json.dumps(dict(
        python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
        queries_sha256=hashlib.sha256(Q.read_bytes()).hexdigest(), k=10, split="test_only", device="cpu"), indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
