"""Phase 6 diagnostic : the Phase 5 checker verifies per-item facts but not
group-level claims. The audit showed these are the main undetected errors, so this script counts them with simple
rules on the saved eval outputs (arm C and D) and on the 45 audited answers.

Rules (fixed once, text only):
  ordering   a sentence with 'highest' ... 'followed by' lists xG values A, B, C in that order; error if not A >= B >= C
  all_goals  'all three ... goal(s)' without 'none'/'no goals' but not all 3 evidence outcomes are goals (same for saved/blocked)
  all_foot   'all ... right-footed'/'left-footed' but a cited-evidence item uses the other foot (or a head)
  all_area   'all ... inside the box' / 'outside the box' but an evidence item is in the other area
Outputs: results/phase6/group_claims.csv (per answer) and group_claims_summary.json."""
import csv, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/phase6"
NEG = re.compile(r"\b(none|no goals?|not goals?|not a goal|zero)\b", re.I)
OUTC = {"goal": "goal", "goals": "goal", "saved": "saved", "blocked": "blocked"}


def check(ex, ans):
    ev = ex["evidence"]
    errs = []
    if len(ev) < 3:
        return errs
    low = ans.lower()
    for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z\[])", ans):
        s = sent.lower()
        if "highest" in s and "followed by" in s:
            tail = s[s.index("highest"):]
            vals = [float(v) for v in re.findall(r"\b(\d\.\d{1,2}|0\.\d+)\b", tail) if float(v) < 1.0]
            if len(vals) >= 2 and any(vals[i] < vals[i + 1] - 1e-9 for i in range(len(vals) - 1)):
                errs.append("ordering")
        m = re.search(r"\ball (?:three )?(?:of these |of the )?[^.]*?\b(goals?|saved|blocked)\b", s)
        if m and not NEG.search(s) and "one goal" not in s:
            tgt = OUTC[m.group(1)]
            if any(e["outcome"] != tgt for e in ev):
                errs.append("all_" + tgt)
        for foot, key in (("right-footed", "right foot"), ("left-footed", "left foot")):
            if re.search(rf"\ball [^.]*?{foot}", s) and any(e["doc_attrs"]["body"] != key for e in ev):
                errs.append("all_foot")
        for area in ("inside the box", "outside the box"):
            if re.search(rf"\ball [^.]*?\b(?:came )?(?:from )?{area}", s) and any(e["doc_attrs"]["area"] != area for e in ev):
                errs.append("all_area")
    return sorted(set(errs))


def main():
    ex = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "data/processed/gen_inputs_eval.jsonl").read_text(encoding="utf-8").splitlines()}
    outs = [json.loads(l) for l in (OUT.parent / "phase5/eval_outputs.jsonl").read_text(encoding="utf-8").splitlines()]
    d = {json.loads(l)["id"]: json.loads(l)["answer"] for l in (OUT.parent / "phase5/eval_outputs_D.jsonl").read_text(encoding="utf-8").splitlines()}
    rows = []
    for o in outs:
        if o["arm"] == "C":
            rows.append(("C", o["id"], o["passed"], check(ex[o["id"]], o["answer"])))
    for i, a in d.items():
        rows.append(("D", i, True, check(ex[i], a)))
    with open(OUT / "group_claims.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["arm", "id", "checker_passed", "group_claim_errors"])
        for r in rows:
            w.writerow([r[0], r[1], int(r[2]), ";".join(r[3])])
    summ = {}
    for arm in "CD":
        sub = [r for r in rows if r[0] == arm and ex[r[1]]["evidence"]]
        bad = [r for r in sub if r[3]]
        summ[arm] = dict(n_with_evidence=len(sub), with_group_error=len(bad),
                         among_checker_passed=sum(1 for r in bad if r[2]), n_checker_passed=sum(1 for r in sub if r[2]),
                         by_type={t: sum(t in r[3] or any(x.startswith(t) for x in r[3]) for r in bad) for t in ("ordering", "all_goal", "all_saved", "all_blocked", "all_foot", "all_area")})
    # validation against the AI audit labels
    lab = {r["id"]: r for r in csv.DictReader(open(OUT.parent / "phase5/audit_sheet.csv", encoding="utf-8-sig"))}
    cmap = {r[1]: r[3] for r in rows if r[0] == "C"}
    n_unf = [i for i, r in lab.items() if r["faithful_YN"] == "N"]
    summ["vs_audit"] = dict(n_audited=len(lab), unfaithful_labelled=len(n_unf), of_which_rule_flags=sum(bool(cmap.get(i)) for i in n_unf),
                            faithful_labelled=len(lab) - len(n_unf), faithful_but_rule_flags=sum(bool(cmap.get(i)) for i, r in lab.items() if r["faithful_YN"] == "Y"))
    # the 50 independent questions (arm C), same rules
    ex50 = {json.loads(l)["id"]: json.loads(l) for l in (OUT / "queries_50_inputs.jsonl").read_text(encoding="utf-8").splitlines()}
    g50 = [json.loads(l) for l in (OUT / "gen50_outputs.jsonl").read_text(encoding="utf-8").splitlines()]
    c50 = [o for o in g50 if o["arm"] == "C" and ex50[o["id"]]["evidence"]]
    bad50 = [o for o in c50 if check(ex50[o["id"]], o["answer"])]
    summ["gen50_C"] = dict(n_with_evidence=len(c50), with_group_error=len(bad50), among_checker_passed=sum(1 for o in bad50 if o["strict_pass"]),
                           n_checker_passed=sum(1 for o in c50 if o["strict_pass"]))
    (OUT / "group_claims_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
