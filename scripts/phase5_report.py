"""Phase 5 reporting: re-scores the SAVED eval outputs (no generation, no model, checker and decoding unchanged).

Writes results/phase5/eval_components.csv, eval_components_bootstrap_C_vs_D.csv, eval_by_type.csv,
eval_outputs_D.jsonl, audit_sheet.csv, audit_key.csv.  Nothing here is used to choose anything.

Arm D (rule-based template, written once, not tuned to the checker; uses only fields the model also sees):
  * no evidence in the input -> two refusal sentences (names the unsupported part, minute / player name, if the
    retrieval flagged one, and lists the five supported attributes);
  * otherwise one sentence per evidence item "[i] <team> <body> shot from <area>, <side>: <outcome> (xG <x.xx>)."
    plus a last sentence "Mean xG of the three is <m>; <g> of 3 were goals." (+ "; I can't search by <part>, so that
    part of the question was ignored" when the retrieval flagged one).

Component definitions (all computed with grounding.check_answer on the evidence block the model received):
  strict_pass        checker `passed` (all 143)
  xg_correct         >=1 xG value stated and every stated value matches its item or the mean (evidence queries, n=130)
  outcome_correct    no outcome-word error (evidence queries, n=130; vacuously true if no outcome is stated)
  unsupported_mean   mean number of checker-listed unsupported claims per answer (all 143)
  unsupported_share  share of answers with >=1 unsupported claim (all 143)
  citation_present   >=1 [n] marker and all markers in range (evidence queries, n=130)
  correct_refusal    on the 13 refuse-mode queries: the answer contains a refusal phrase (REFUSAL_RE), has no [n]
                     citation, states no xG value and has no unsupported claim (n=13)
  false_refusal      (extra) evidence queries whose answer contains REFUSAL_RE and no [n] citation (n=130)
"""
import csv, json, random, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from football_rag.gen_format import SUPPORTED_ATTRIBUTES, fold_text  # noqa: E402
from football_rag.grounding import check_answer  # noqa: E402

OUT = ROOT / "results/phase5"
SEED_BOOT, SEED_AUDIT, N_BOOT = 20261021, 7, 10_000
REFUSAL_RE = re.compile(r"\b(cannot|can't|can not|unable|not able|isn't supported|aren't supported|not supported|"
                        r"outside what|outside my|i only (answer|cover|work|search))\b", re.I)
CITE = re.compile(r"\[(\d)\]")
COMPONENTS = ["strict_pass", "xg_correct", "outcome_correct", "unsupported_mean", "unsupported_share",
              "citation_present", "correct_refusal", "false_refusal"]


def arm_d(ex):
    ev, notes = ex["evidence"], ex["retrieval"]["unsupported"]
    part = " and ".join(p for p, cue in (("minute", "minute"), ("player name", "player")) if any(cue in n for n in notes))
    if not ev:
        first = f"I can't search by {part}." if part else "I only answer questions about football shots."
        return f"{first} You can ask about a shot's team, body part, side, area or outcome."
    sents = [f"[{i}] {fold_text(e['team'])} {e['doc_attrs']['body']} shot from {e['doc_attrs']['area']}, "
             f"{e['doc_attrs']['side']}: {e['outcome']} (xG {e['xg']:.2f})." for i, e in enumerate(ev, 1)]
    last = f"Mean xG of the three is {ex['summary']['mean_xg']:.2f}; {sum(e['goal'] for e in ev)} of 3 were goals"
    last += f"; I can't search by {part}, so that part of the question was ignored." if part else "."
    return " ".join(sents + [last])


def per_query(ex, answer):
    r = check_answer(ex, answer)
    ev_mode = ex["mode"] in ("answer", "partial")
    cites = CITE.findall(answer)
    refused = bool(REFUSAL_RE.search(answer))
    d = dict(id=ex["id"], kind=ex["kind"], mode=ex["mode"], strict_pass=int(r["passed"]),
             unsupported_n=len(r["unsupported"]), unsupported_share=int(bool(r["unsupported"])))
    d["xg_correct"] = int(bool(r["xg_details"]) and r["xg_ok"]) if ev_mode else None
    d["outcome_correct"] = int(r["outcome_ok"]) if ev_mode else None
    d["citation_present"] = int(bool(cites) and r["citations_ok"]) if ev_mode else None
    d["false_refusal"] = int(refused and not cites) if ev_mode else None
    d["correct_refusal"] = (int(refused and not cites and not r["xg_details"] and not r["unsupported"])
                            if ex["mode"] == "refuse" else None)
    d["unsupported_mean"] = d["unsupported_n"]
    return d


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def rate(rows, comp):
    v = [r[comp] for r in rows if r[comp] is not None]
    return (sum(v) / len(v)) if v else float("nan"), len(v), sum(v)


def main():
    ex = [json.loads(l) for l in (ROOT / "data/processed/gen_inputs_eval.jsonl").read_text(encoding="utf-8").splitlines()]
    saved = defaultdict(dict)
    for l in (OUT / "eval_outputs.jsonl").read_text(encoding="utf-8").splitlines():
        o = json.loads(l); saved[o["arm"]][o["id"]] = o["answer"]
    saved["D"] = {e["id"]: arm_d(e) for e in ex}
    (OUT / "eval_outputs_D.jsonl").write_text("\n".join(json.dumps(dict(arm="D", id=i, answer=a), ensure_ascii=False)
                                                       for i, a in saved["D"].items()), encoding="utf-8")
    rows = {arm: [per_query(e, saved[arm][e["id"]]) for e in ex] for arm in "ABCD"}

    with open(OUT / "eval_components.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["arm", "component", "n", "k", "value", "wilson95_lo", "wilson95_hi"])
        for arm in "ABCD":
            for c in COMPONENTS:
                if c == "unsupported_mean":
                    v = [r[c] for r in rows[arm]]; w.writerow([arm, c, len(v), "", f"{np.mean(v):.4f}", "", ""]); continue
                p, n, k = rate(rows[arm], c); lo, hi = wilson(k, n)
                w.writerow([arm, c, n, k, f"{p:.4f}", f"{lo:.4f}", f"{hi:.4f}"])

    rng = np.random.default_rng(SEED_BOOT); N = len(ex)
    idx = rng.integers(0, N, size=(N_BOOT, N))
    with open(OUT / "eval_components_bootstrap_C_vs_D.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["component", "C", "D", "diff_C_minus_D", "boot95_lo", "boot95_hi", "n_resamples"])
        for c in COMPONENTS:
            vc = np.array([np.nan if r[c] is None else r[c] for r in rows["C"]], float)
            vd = np.array([np.nan if r[c] is None else r[c] for r in rows["D"]], float)
            def mean_of(v):
                s = v[idx]; return np.nanmean(s, axis=1)
            with np.errstate(all="ignore"):
                diffs = mean_of(vc) - mean_of(vd)
            lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
            w.writerow([c, f"{np.nanmean(vc):.4f}", f"{np.nanmean(vd):.4f}", f"{np.nanmean(vc) - np.nanmean(vd):+.4f}",
                        f"{lo:+.4f}", f"{hi:+.4f}", N_BOOT])

    with open(OUT / "eval_by_type.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["arm", "kind", "n"] + COMPONENTS)
        for arm in "ABCD":
            for k in sorted({r["kind"] for r in rows[arm]}):
                sub = [r for r in rows[arm] if r["kind"] == k]
                vals = []
                for c in COMPONENTS:
                    v = [r[c] for r in sub if r[c] is not None]
                    vals.append("" if not v else f"{np.mean(v):.3f}")
                w.writerow([arm, k, len(sub)] + vals)

    # audit sheet: 30 random checker-passing C answers + all flagged C answers, shuffled, verdict hidden
    byid = {e["id"]: e for e in ex}
    passed = [r["id"] for r in rows["C"] if r["strict_pass"]]
    flagged = [r["id"] for r in rows["C"] if not r["strict_pass"]]
    rnd = random.Random(SEED_AUDIT)
    pick = rnd.sample(passed, 30) + flagged
    rnd.shuffle(pick)
    key = {}
    with open(OUT / "audit_sheet.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["id", "question", "evidence_1", "evidence_2", "evidence_3", "answer",
                                       "faithful_YN", "clarity_1to5", "notes"])
        for i in pick:
            e = byid[i]
            lines = re.findall(r"^\[\d\] .*$", e["input"], re.M)
            lines = [re.sub(r" \| split: \w+$", "", s) for s in lines] + [""] * 3
            w.writerow([i, e["query"], *lines[:3], saved["C"][i], "", "", ""])
            res = check_answer(e, saved["C"][i])
            key[i] = (int(res["passed"]), "; ".join(res["reasons"]))
    with open(OUT / "audit_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "checker_passed", "checker_reasons"])
        for i in pick:
            w.writerow([i, *key[i]])
    print("audit rows", len(pick), "flagged", len(flagged), "passed-sampled", len(pick) - len(flagged))


if __name__ == "__main__":
    main()
