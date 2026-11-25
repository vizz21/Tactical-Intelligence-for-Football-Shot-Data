"""Phase 5 arm C: fine-tune flan-t5-base on gen_targets_train, select the epoch by VAL loss, then generate and
score on VAL with the same decoding and checker as arms A/B. Never reads the eval file.

Fixed before running: AdamW lr 3e-4 (linear decay, 5% warmup), effective batch 8 (4 x 2), bf16 autocast with fp32
weights, up to 8 epochs, keep the epoch with the lowest val loss, stop after 2 epochs without improvement.
Outputs: models/flan_t5_armC/ (best checkpoint), results/phase5/armC_{train_log.json,val_outputs.jsonl,
val_summary.json,run_info.json}. Progress is printed per 25 steps.
"""
import hashlib, json, platform, random, sys, time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from phase5_zeroshot_val import GEN, MODEL, SEED, run, score, summarise, sha  # noqa: E402

D = ROOT / "data/processed"
OUT = ROOT / "results/phase5"
CKPT = ROOT / "models/flan_t5_armC"
LR, EPOCHS, PATIENCE, BS, ACCUM, MAX_IN, MAX_OUT = 3e-4, 8, 2, 4, 2, 512, 192


def load(split):
    tg = [json.loads(l) for l in (D / f"gen_targets_{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    inp = {r["id"]: r for r in (json.loads(l) for l in (D / f"gen_inputs_{split}.jsonl").read_text(encoding="utf-8").splitlines())}
    return tg, inp


def encode(tok, recs):
    x = tok([r["input"] for r in recs], max_length=MAX_IN, truncation=True)
    y = tok([r["target"] for r in recs], max_length=MAX_OUT, truncation=True)
    assert max(len(i) for i in x["input_ids"]) <= MAX_IN
    n_trunc = sum(len(t) >= MAX_OUT for t in y["input_ids"])
    assert n_trunc == 0, f"{n_trunc} targets hit the length cap"
    return list(zip(x["input_ids"], y["input_ids"]))


def batches(pairs, tok, bs, shuffle, rng):
    idx = list(range(len(pairs)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [pairs[j] for j in idx[i:i + bs]]
        li, lo = max(len(a) for a, _ in chunk), max(len(b) for _, b in chunk)
        ids = torch.tensor([a + [tok.pad_token_id] * (li - len(a)) for a, _ in chunk])
        lab = torch.tensor([b + [-100] * (lo - len(b)) for _, b in chunk])
        yield ids, (ids != tok.pad_token_id).long(), lab


@torch.no_grad()
def val_loss(model, pairs, tok, device):
    model.eval(); tot = n = 0
    for ids, att, lab in batches(pairs, tok, 8, False, None):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out = model(input_ids=ids.to(device), attention_mask=att.to(device), labels=lab.to(device))
        k = (lab != -100).sum().item(); tot += out.loss.float().item() * k; n += k
    return tot / n


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    rng = random.Random(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(device)
    tr, _ = load("train"); va, va_in = load("val")
    ptr, pva = encode(tok, tr), encode(tok, va)
    print(f"device {device}; train {len(ptr)} val {len(pva)}", flush=True)

    steps = EPOCHS * ((len(ptr) + BS * ACCUM - 1) // (BS * ACCUM))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    log, best, bad, t0 = [], 1e9, 0, time.time()
    for ep in range(1, EPOCHS + 1):
        model.train(); losses = []; opt.zero_grad()
        for i, (ids, att, lab) in enumerate(batches(ptr, tok, BS, True, rng), 1):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(input_ids=ids.to(device), attention_mask=att.to(device), labels=lab.to(device)).loss
            (loss / ACCUM).backward(); losses.append(loss.item())
            if i % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad()
            if i % 50 == 0:
                print(f"ep {ep} batch {i} loss {np.mean(losses[-50:]):.4f} t {time.time() - t0:.0f}s", flush=True)
        vl = val_loss(model, pva, tok, device)
        log.append(dict(epoch=ep, train_loss=float(np.mean(losses)), val_loss=vl, elapsed_s=time.time() - t0))
        print(f"== epoch {ep}: train {log[-1]['train_loss']:.4f} val {vl:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if vl < best:
            best, bad = vl, 0
            model.save_pretrained(CKPT); tok.save_pretrained(CKPT)
        else:
            bad += 1
            if bad >= PATIENCE:
                print("early stop", flush=True); break
    best_ep = min(log, key=lambda r: r["val_loss"])["epoch"]

    model = AutoModelForSeq2SeqLM.from_pretrained(CKPT).to(device).eval()
    rows = [va_in[r["id"]] for r in va]
    torch.manual_seed(SEED)
    ans = run(model, tok, [r["input"] for r in rows], device)
    per = score(rows, ans)
    summ = summarise(per)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "armC_train_log.json").write_text(json.dumps(dict(best_epoch=best_ep, log=log), indent=1), encoding="utf-8")
    (OUT / "armC_val_outputs.jsonl").write_text("\n".join(json.dumps(dict(arm="C", **p), ensure_ascii=False) for p in per), encoding="utf-8")
    (OUT / "armC_val_summary.json").write_text(json.dumps(dict(split="val", model=MODEL, decoding=GEN, best_epoch=best_ep, results=summ), indent=1), encoding="utf-8")
    (OUT / "armC_run_info.json").write_text(json.dumps(dict(
        seed=SEED, model=MODEL, lr=LR, epochs_max=EPOCHS, patience=PATIENCE, batch=BS, accum=ACCUM, precision="bf16 autocast, fp32 weights",
        device=device, python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
        files=dict(train_targets=sha(D / "gen_targets_train.jsonl"), val_targets=sha(D / "gen_targets_val.jsonl"),
                   val_inputs=sha(D / "gen_inputs_val.jsonl"), grounding_py=sha(ROOT / "src/football_rag/grounding.py"), script=sha(__file__),
                   checkpoint_weights=sha(CKPT / "model.safetensors")), eval_file_read=False), indent=1), encoding="utf-8")
    a = summ["all"]
    print(f"VAL arm C: strict {a['strict_pass']:.3f} grounding-only {a['grounding_only_pass']:.3f} refusal {a['refusal_rate']:.3f}", flush=True)


if __name__ == "__main__":
    main()
