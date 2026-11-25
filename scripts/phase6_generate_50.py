"""Phase 6 step 4: arm C (and rule-based D for reference) on the 50 independent questions, test-split evidence.
Same decoding and checker as Phase 5; nothing is tuned. Input: results/phase6/queries_50_inputs.jsonl (from
phase6_query_eval.py). Outputs: results/phase6/gen50_{outputs.jsonl,summary.json,run_info.json}."""
import csv, hashlib, json, platform, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from phase5_report import REFUSAL_RE, arm_d, per_query  # noqa: E402
from phase5_zeroshot_val import GEN, SEED, run  # noqa: E402

OUT = ROOT / "results/phase6"
CKPT = ROOT / "models/flan_t5_armC"


def main():
    rows = [json.loads(l) for l in (OUT / "queries_50_inputs.jsonl").read_text(encoding="utf-8").splitlines()]
    meta = {r["id"]: r for r in csv.DictReader(open(ROOT / "data/phase6/queries_50.csv", encoding="utf-8"))}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForSeq2SeqLM.from_pretrained(CKPT).to(device).eval()
    torch.manual_seed(SEED)
    ans_c = run(model, tok, [r["input"] for r in rows], device)
    outs, summ = [], {}
    for arm, answers in (("C", ans_c), ("D", [arm_d(r) for r in rows])):
        per = []
        for r, a in zip(rows, answers):
            d = per_query(r, a)
            m = meta[r["id"]]
            d.update(arm=arm, answer=a, category=m["category"], expected_scope=m["expected_scope"], system_mode=r["mode"],
                     refused_text=int(bool(REFUSAL_RE.search(a))), reasons=check_reasons(r, a))
            per.append(d)
        outs += per
        summ[arm] = dict(
            n=len(per), strict_pass=sum(d["strict_pass"] for d in per),
            by_category={c: dict(n=len([d for d in per if d["category"] == c]), strict_pass=sum(d["strict_pass"] for d in per if d["category"] == c))
                         for c in sorted({d["category"] for d in per})},
            flags=dict(Counter(f for d in per for f in d["reasons"]["flags"])),
            # expected out of scope but the pipeline still gave evidence: did the text refuse?
            out_of_scope_expected=dict(n=sum(d["expected_scope"] == "out" for d in per),
                                       text_refused=sum(d["expected_scope"] == "out" and d["refused_text"] == 1 and not d["answer"].count("[") for d in per)),
            in_scope_refused=sum(d["expected_scope"] == "in" and d["refused_text"] == 1 and "[" not in d["answer"] for d in per))
    (OUT / "gen50_outputs.jsonl").write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in outs), encoding="utf-8")
    (OUT / "gen50_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    (OUT / "gen50_run_info.json").write_text(json.dumps(dict(
        seed=SEED, decoding=GEN, device=device, torch=torch.__version__, transformers=transformers.__version__, python=platform.python_version(),
        inputs_sha256=hashlib.sha256((OUT / "queries_50_inputs.jsonl").read_bytes()).hexdigest(),
        armC_weights=hashlib.sha256((CKPT / "model.safetensors").read_bytes()).hexdigest()), indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


def check_reasons(ex, answer):
    from football_rag.grounding import check_answer
    r = check_answer(ex, answer)
    flags = []
    if not r["xg_ok"]: flags.append("xg")
    if not r["outcome_ok"]: flags.append("outcome")
    flags += sorted({"unsupported_" + u["type"] for u in r["unsupported"]})
    if not r["citations_ok"]: flags.append("citation")
    if not r["format_ok"]: flags.append("format")
    return dict(flags=flags, text=r["reasons"])


if __name__ == "__main__":
    main()
