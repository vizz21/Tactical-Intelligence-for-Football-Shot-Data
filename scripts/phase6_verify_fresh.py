"""Phase 6: test verify-and-fall-back on NEW questions (data/phase6/queries_fresh60.csv, written after the verifier
rules were frozen; hashes in data/phase6/*.sha256 and results/phase6/verify_py.sha256).

For each question: pipeline (test split, k=3) -> arm C answer -> verify.choose_answer. Reports raw C vs the final
answer on the Phase 5 checker, the fallback rate and its reasons. Writes a blind audit sheet of the answers that
were KEPT from the model (source hidden) so the remaining error rate can be measured independently of the rules.
Outputs: results/phase6/verify_fresh_{outputs.jsonl,summary.json,run_info.json}, audit_sheet_fresh.csv, audit_key_fresh.csv.
"""
import csv, hashlib, json, platform, random, re, sys
from collections import Counter
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from football_rag.grounding import check_answer  # noqa: E402
from football_rag.verify import choose_answer, group_errors  # noqa: E402
from phase5_build_inputs import Builder  # noqa: E402
from phase5_zeroshot_val import GEN, SEED, run  # noqa: E402

Q = ROOT / "data/phase6/queries_fresh60.csv"
OUT = ROOT / "results/phase6"
CKPT = ROOT / "models/flan_t5_armC"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731


def main():
    assert sha(Q) == (ROOT / "data/phase6/queries_fresh60.sha256").read_text().split()[0], "query file changed"
    assert sha(ROOT / "src/football_rag/verify.py") == (OUT / "verify_py.sha256").read_text().split()[0], "verifier changed after freezing"
    rows = list(csv.DictReader(open(Q, encoding="utf-8")))
    b = Builder()
    ex = [b.example(r["id"], "test", "fresh60", r["question"], None) for r in rows]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForSeq2SeqLM.from_pretrained(CKPT).to(device).eval()
    torch.manual_seed(SEED)
    raw = run(model, tok, [e["input"] for e in ex], device)
    outs = []
    for r, e, a in zip(rows, ex, raw):
        text, src, why = choose_answer(e, a)
        c_raw, c_fin = check_answer(e, a), check_answer(e, text)
        outs.append(dict(id=r["id"], question=r["question"], expected_scope=r["expected_scope"], system_mode=e["mode"], raw=a, final=text,
                         source=src, reasons=why, raw_passed=c_raw["passed"], final_passed=c_fin["passed"],
                         raw_group_errors=group_errors(e, a), final_group_errors=group_errors(e, text)))
    (OUT / "verify_fresh_outputs.jsonl").write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in outs), encoding="utf-8")
    ev = [o for o in outs if o["system_mode"] != "refuse"]
    summ = dict(n=len(outs), n_with_evidence=len(ev),
                raw_strict_pass=sum(o["raw_passed"] for o in outs), final_strict_pass=sum(o["final_passed"] for o in outs),
                raw_with_group_error=sum(bool(o["raw_group_errors"]) for o in ev), final_with_group_error=sum(bool(o["final_group_errors"]) for o in outs),
                fallback=sum(o["source"] == "template" for o in outs), fallback_with_evidence=sum(o["source"] == "template" for o in ev),
                kept_from_model=sum(o["source"] == "model" for o in outs),
                fallback_reasons=dict(Counter(re.sub(r"[:'].*", "", x) if not x.startswith("group:") else x for o in outs for x in o["reasons"])))
    (OUT / "verify_fresh_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    (OUT / "verify_fresh_run_info.json").write_text(json.dumps(dict(
        seed=SEED, decoding=GEN, device=device, torch=torch.__version__, transformers=transformers.__version__, python=platform.python_version(),
        queries_sha256=sha(Q), verify_py_sha256=sha(ROOT / "src/football_rag/verify.py"), armC_weights=sha(CKPT / "model.safetensors")), indent=1), encoding="utf-8")
    # blind audit sheet: answers kept from the model that contain evidence
    kept = [o for o in outs if o["source"] == "model" and next(e for e in ex if e["id"] == o["id"])["evidence"]]
    random.Random(7).shuffle(kept)
    byid = {e["id"]: e for e in ex}
    with open(OUT / "audit_sheet_fresh.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["id", "question", "evidence_1", "evidence_2", "evidence_3", "answer", "faithful_YN", "clarity_1to5", "notes"])
        for o in kept:
            lines = re.findall(r"^\[\d\] .*$", byid[o["id"]]["input"], re.M)
            lines = [re.sub(r" \| split: \w+$", "", s) for s in lines] + [""] * 3
            w.writerow([o["id"], o["question"], *lines[:3], o["final"], "", "", ""])
    print(json.dumps(summ, indent=1)); print("audit rows", len(kept))


if __name__ == "__main__":
    main()
