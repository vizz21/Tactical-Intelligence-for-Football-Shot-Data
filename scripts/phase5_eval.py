"""Phase 5 final eval: arms A, B (wording B3, chosen on val) and C (models/flan_t5_armC, epoch chosen on val)
on the fixed eval set (143 queries). No choices are made here; run once. Refuses to overwrite earlier eval outputs.
Outputs: results/phase5/eval_{outputs.jsonl,summary.json,run_info.json}.
"""
import json, platform, sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from phase5_zeroshot_val import GEN, MODEL, PREFIXES, SEED, arm_a_input, run, score, sha, summarise  # noqa: E402

EVAL = ROOT / "data/processed/gen_inputs_eval.jsonl"
OUT = ROOT / "results/phase5"
CKPT = ROOT / "models/flan_t5_armC"


def main():
    assert not (OUT / "eval_summary.json").exists(), "eval already run once; do not repeat"
    expected = (ROOT / "data/processed/gen_inputs_eval.sha256").read_text().split()[0]
    assert sha(EVAL) == expected, "eval file changed"
    sel = json.loads((OUT / "zeroshot_val_summary.json").read_text())["selected_B"]
    rows = [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    arms = {"A": ("base", [arm_a_input(r) for r in rows]),
            "B": ("base", [PREFIXES[sel] + r["input"] for r in rows]),
            "C": ("armC", [r["input"] for r in rows])}
    models = {"base": MODEL, "armC": str(CKPT)}
    outputs, results = [], {}
    for name, (m, texts) in arms.items():
        model = AutoModelForSeq2SeqLM.from_pretrained(models[m]).to(device).eval()
        torch.manual_seed(SEED)
        per = score(rows, run(model, tok, texts, device))
        outputs += [dict(arm=name, **p) for p in per]
        res = summarise(per)
        res["by_kind"] = {}
        for k in sorted({r["kind"] for r in rows}):
            sub = [p for p in per if p["kind"] == k]
            res["by_kind"][k] = dict(n=len(sub), strict_pass=sum(p["passed"] for p in sub) / len(sub),
                                     flag_counts=dict(Counter(f for p in sub for f in p["flags"])))
        results[name] = res
        a = res["all"]
        print(name, f"strict {a['strict_pass']:.3f} grounding-only {a['grounding_only_pass']:.3f}", flush=True)
        del model; torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eval_outputs.jsonl").write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in outputs), encoding="utf-8")
    (OUT / "eval_summary.json").write_text(json.dumps(dict(split="eval", n=len(rows), decoding=GEN, B_wording=sel, results=results), indent=1), encoding="utf-8")
    (OUT / "eval_run_info.json").write_text(json.dumps(dict(
        seed=SEED, model=MODEL, decoding=GEN, device=device, python=platform.python_version(), torch=torch.__version__,
        transformers=transformers.__version__, numpy=np.__version__,
        files=dict(eval_inputs=sha(EVAL), grounding_py=sha(ROOT / "src/football_rag/grounding.py"), script=sha(__file__),
                   armC_weights=sha(CKPT / "model.safetensors"))), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
