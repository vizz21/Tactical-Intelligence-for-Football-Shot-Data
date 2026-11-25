"""Phase 5 arms A and B: flan-t5-base inference only (no fine-tuning), VAL split only.

Arm A: question only (evidence and summary stripped).  Arm B: instruction prefix + full input; 3 wordings tried.
Selection rule (fixed before running): best wording = highest strict pass rate on val, ties broken by the
'grounding-only' pass rate (strict minus the format check), then by fewer flags.
Writes results/phase5/zeroshot_val_{outputs.jsonl,summary.json}, run_info.json. Never reads the eval file.
"""
import hashlib, json, platform, random, re, sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from football_rag.grounding import check_answer  # noqa: E402

SEED = 42
MODEL = "google/flan-t5-base"
GEN = dict(num_beams=4, max_new_tokens=128, no_repeat_ngram_size=3, do_sample=False)
VAL = ROOT / "data/processed/gen_inputs_val.jsonl"
OUT = ROOT / "results/phase5"

PREFIXES = {
    "B1": "Answer the question using only the evidence below. Cite the evidence as [1], [2], [3]. "
          "If the question cannot be answered from the evidence, say so.\n\n",
    "B2": "You are a football shot assistant. Write 2 to 4 sentences. Use only facts that appear in the evidence: "
          "copy the xG values exactly, give the outcome as stated, and cite each shot with [1], [2] or [3]. "
          "If there is no evidence or the question is not about football shots, refuse and list what you can search "
          "by: team, body part, side, area and outcome.\n\n",
    "B3": "Task: summarise the evidence for the question. Rules: do not add any player, team, minute or number that is "
          "not in the evidence; cite every shot as [n]; if the evidence is missing or does not match the question, "
          "reply that you cannot answer.\n\n",
}
REFUSE_RE = re.compile(r"\b(cannot|can't|can not|unable|not able|no evidence|not enough|don't know|do not know|"
                       r"not possible|i'm sorry|sorry|not supported|isn't supported|outside)\b", re.I)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def arm_a_input(ex):
    return "Question: " + ex["query"]


def run(model, tok, texts, device, bs=8):
    outs = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=640).to(device)
        with torch.no_grad():
            g = model.generate(**enc, **GEN)
        outs += tok.batch_decode(g, skip_special_tokens=True)
    return outs


def score(rows, answers):
    per = []
    for ex, ans in zip(rows, answers):
        r = check_answer(ex, ans)
        grounding_only = r["xg_ok"] and r["outcome_ok"] and not r["unsupported"] and r["citations_ok"]
        refused = bool(REFUSE_RE.search(ans))
        flags = []
        if not r["xg_ok"]:
            flags.append("xg")
        if not r["outcome_ok"]:
            flags.append("outcome")
        if r["unsupported"]:
            flags += sorted({"unsupported_" + u["type"] for u in r["unsupported"]})
        if not r["citations_ok"]:
            flags.append("citation")
        if not r["format_ok"]:
            flags.append("format")
        per.append(dict(id=ex["id"], mode=ex["mode"], kind=ex["kind"], answer=ans, passed=r["passed"],
                        grounding_only=grounding_only, refused=refused, flags=flags, reasons=r["reasons"]))
    return per


def summarise(per):
    def block(sub):
        n = len(sub)
        if not n:
            return None
        fl = Counter(f for p in sub for f in p["flags"])
        return dict(n=n, strict_pass=sum(p["passed"] for p in sub) / n,
                    grounding_only_pass=sum(p["grounding_only"] for p in sub) / n,
                    refusal_rate=sum(p["refused"] for p in sub) / n,
                    has_citation=sum(bool(re.search(r"\[[123]\]", p["answer"])) for p in sub) / n,
                    flag_counts=dict(fl))
    s = {"all": block(per)}
    for m in ("answer", "partial", "refuse"):
        s[m] = block([p for p in per if p["mode"] == m])
    s["supported_answer_or_partial"] = block([p for p in per if p["mode"] in ("answer", "partial")])
    return s


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    rows = [json.loads(l) for l in VAL.read_text(encoding="utf-8").splitlines()]
    assert all(r["pool"] == "val" or r["pool"] == "val" for r in rows) and len(rows) == 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(device).eval()

    results, outputs = {}, []
    arms = {"A": [arm_a_input(r) for r in rows]}
    for k, pre in PREFIXES.items():
        arms[k] = [pre + r["input"] for r in rows]
    for name, texts in arms.items():
        torch.manual_seed(SEED)
        ans = run(model, tok, texts, device)
        per = score(rows, ans)
        results[name] = summarise(per)
        for p in per:
            outputs.append(dict(arm=name, **p))
        a = results[name]["all"]
        print(name, f"strict {a['strict_pass']:.2f} grounding-only {a['grounding_only_pass']:.2f} refusal {a['refusal_rate']:.2f}", flush=True)

    best = max(PREFIXES, key=lambda k: (results[k]["all"]["strict_pass"], results[k]["all"]["grounding_only_pass"],
                                         -sum(results[k]["all"]["flag_counts"].values())))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "zeroshot_val_outputs.jsonl").write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in outputs), encoding="utf-8")
    (OUT / "zeroshot_val_summary.json").write_text(json.dumps(
        dict(split="val", model=MODEL, decoding=GEN, prefixes=PREFIXES, selected_B=best,
             selection_rule="strict pass, then grounding-only pass, then fewer flags (val only)", results=results),
        indent=1, ensure_ascii=False), encoding="utf-8")
    (OUT / "zeroshot_run_info.json").write_text(json.dumps(dict(
        seed=SEED, model=MODEL, decoding=GEN, device=device, python=platform.python_version(),
        torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
        files=dict(val_inputs=sha(VAL), grounding_py=sha(ROOT / "src/football_rag/grounding.py"),
                   gen_format_py=sha(ROOT / "src/football_rag/gen_format.py"), script=sha(__file__)),
        eval_file_read=False), indent=1), encoding="utf-8")
    print("selected B:", best)


if __name__ == "__main__":
    main()
