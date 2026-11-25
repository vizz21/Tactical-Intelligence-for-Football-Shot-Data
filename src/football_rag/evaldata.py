"""Data for the page's Evaluation tab, read from the saved result files (nothing is recomputed or retrained here).

`load_evaluation()` returns {"sections": [...], "cautions": [...]}. A section is either
  {"id", "title", "note", "kind": "table", "columns": [...], "rows": [[...], ...]} or
  {"id", "title", "note", "kind": "bars", "items": [{"label", "value" (0-1), "lo", "hi", "detail"}]}.
A section whose source file is missing is skipped. The retrieval baseline table is typed in from
claude/phase3_bi_encoder_results.md (the notebook output is not stored as a machine-readable file).
"""
import csv
import json
import math
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def _csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _item(label: str, k: int, n: int, detail: str = "") -> dict:
    lo, hi = wilson(k, n)
    return {"label": label, "value": k / n if n else 0.0, "lo": lo, "hi": hi, "detail": f"{k}/{n}" + (f". {detail}" if detail else "")}


def _xg():
    p = R / "phase2_script" / "phase2_results.csv"
    if not p.exists():
        return []
    rows = [[r["model"], r["test_log_loss"], r["test_auc"], r["test_brier"]] for r in _csv(p)]
    out = [{"id": "xg", "kind": "table", "title": "xG models on the test split (lower log-loss and Brier, higher AUC are better)",
            "note": "ANN and CNN: mean ± seed standard deviation over 5 seeds. The ANN is the xG model used by the app.",
            "columns": ["model", "log-loss", "AUC", "Brier"], "rows": rows}]
    b = R / "phase2_script" / "bootstrap_test.csv"
    if b.exists():
        out.append({"id": "xg_boot", "kind": "table", "title": "Paired match-bootstrap differences (test, 95% CI)",
                    "note": "Every interval includes 0: no reliable difference between the ANN, CNN and logistic regression.",
                    "columns": ["comparison", "metric", "mean difference", "95% CI", "excludes 0"],
                    "rows": [[r["comparison"], r["metric"], r["mean_diff"], f"[{r['ci_low']}, {r['ci_high']}]", r["ci_excludes_0"]] for r in _csv(b)]})
    return out


def _retrieval():
    rows = [["random", "0.071", "0.047", "0.069"], ["BM25", "0.793", "0.327", "0.177"], ["pretrained MiniLM", "0.397", "0.242", "0.292"],
            ["fine-tuned v1", "1.000", "0.600", "0.408"], ["fine-tuned v2 (used by the app)", "1.000", "0.800", "0.654"]]
    out = [{"id": "retr", "kind": "table", "title": "Retrieval nP@10 on the test corpus, by wording",
            "note": "Template queries are saturated (queries and documents share templates), so only near and far paraphrases say anything about robustness. "
                    "Near n=15, far n=13: small, hand-written sets. Source: the Phase 3 retrieval results.",
            "columns": ["system", "template", "near paraphrase", "far paraphrase"], "rows": rows}]
    p = R / "phase6" / "query_eval_summary.json"
    if p.exists():
        s = json.loads(p.read_text(encoding="utf-8"))
        rr = s["retrieval_in_or_partial"]
        rows = []
        for cat, v in s["by_category"].items():
            f = lambda t: "-" if not t or t[0] is None else f"{t[0]:.2f} (n={t[1]})"   # noqa: E731
            rows.append([cat, v["n"], f"{v['scope_correct']}/{v['n']}", f(v["np10"]), f(v["mrr"])])
        rows.append(["all", s["n"], f"{round(s['scope_accuracy'] * s['n'])}/{s['n']}", f"{rr['np10'][0]:.2f} (n={rr['np10'][1]})", f"{rr['mrr'][0]:.2f} (n={rr['mrr'][1]})"])
        out.append({"id": "q50", "kind": "table", "title": "50 independently phrased questions: scope decision and retrieval (test split, k=10)",
                    "note": "Questions written for this test, with intended attributes frozen before the run. Colloquial wording is the weak spot; 3 of 11 out-of-scope questions were still answered.",
                    "columns": ["category", "n", "scope correct", "nP@10 (scoreable)", "MRR (scoreable)"], "rows": rows})
    return out


def _generation():
    p = R / "phase5" / "eval_components.csv"
    if not p.exists():
        return []
    comp = _csv(p)
    names = {"A": "A  question only", "B": "B  zero-shot prompt", "C": "C  fine-tuned FLAN-T5", "D": "D  rule-based template"}
    items = [_item(names[a], int(r["k"]), int(r["n"])) for a in "ABCD" for r in comp if r["arm"] == a and r["component"] == "strict_pass"]
    out = [{"id": "gen", "kind": "bars", "title": "Generation on the fixed 143-question set: strict pass of the automatic checker (95% Wilson interval)",
            "note": "The checker verifies per-item facts. D copies the fields the checker verifies, so 100% is its ceiling, not a better answer.", "items": items}]
    cols = ["component", "A", "B", "C", "D"]
    labels = [("xg_correct", "xG stated and correct (n=130)"), ("outcome_correct", "outcome correct (n=130)"), ("unsupported_share", "answers with an unsupported claim (n=143)"),
              ("citation_present", "citation present (n=130)"), ("correct_refusal", "correct refusal (n=13)")]
    rows = []
    for key, label in labels:
        row = [label]
        for a in "ABCD":
            r = next(x for x in comp if x["arm"] == a and x["component"] == key)
            row.append(f"{float(r['value']) * 100:.1f}%")
        rows.append(row)
    out.append({"id": "gen_comp", "kind": "table", "title": "Generation components", "note": "Outcome-correct is vacuous for A and B, which seldom state an outcome.",
                "columns": cols, "rows": rows})
    t = R / "phase5" / "eval_by_type.csv"
    if t.exists():
        rows = [[r["kind"], r["n"], f"{float(r['strict_pass']) * 100:.0f}%"] for r in _csv(t) if r["arm"] == "C"]
        out.append({"id": "gen_type", "kind": "table", "title": "Arm C by query type (point estimates)", "note": "Groups are small.",
                    "columns": ["query type", "n", "strict pass"], "rows": rows})
    return out


def _audit_counts(sheet: Path, key: Path, key_flag: str):
    lab = {r["id"]: r["faithful_YN"].strip().upper()[:1] for r in _csv(sheet)}
    k = {r["id"]: r for r in _csv(key)}
    return lab, k


def _faithfulness():
    items, notes = [], []
    s5, k5 = R / "phase5" / "audit_sheet.csv", R / "phase5" / "audit_key.csv"
    if s5.exists() and k5.exists() and all(r["faithful_YN"].strip() for r in _csv(s5)):
        lab, key = _audit_counts(s5, k5, "checker_passed")
        passed = [i for i in lab if key[i]["checker_passed"] == "1"]
        bad = sum(lab[i] == "N" for i in passed)
        items.append(_item("Checker-passing C answers that contain an error (audit of 30 answers)", bad, len(passed), "before the fix"))
    g = R / "phase6" / "group_claims_summary.json"
    if g.exists():
        c = json.loads(g.read_text(encoding="utf-8"))["C"]
        items.append(_item("C evidence answers with a group-claim error (text rules, eval)", c["with_group_error"], c["n_with_evidence"], "ranking, 'all three', counts"))
    v = R / "phase6" / "verify_fresh_summary.json"
    if v.exists():
        s = json.loads(v.read_text(encoding="utf-8"))
        items.append(_item("After the fix: evidence answers replaced by the template (60 new questions)", s["fallback_with_evidence"], s["n_with_evidence"], "the model text failed a check"))
    sf, kf = R / "phase6" / "audit_sheet_fresh.csv", R / "phase6" / "audit_key_fresh.csv"
    if sf.exists() and kf.exists():
        lab = {r["id"]: r["faithful_YN"].strip().upper()[:1] for r in _csv(sf)}
        scope = {r["id"]: r["expected_scope"] for r in _csv(kf)}
        ins = [i for i in lab if scope.get(i) == "in"]
        items.append(_item("After the fix: kept model answers with an error, in-scope questions", sum(lab[i] == "N" for i in ins), len(ins), "small sample, wide interval"))
    if not items:
        return []
    return [{"id": "faith", "kind": "bars", "title": "What the checker misses, and the verify-and-fall-back fix",
             "note": "The audit and the new-question ratings are small samples, so read them as indicative. Bars 1, 2 and 4 are error rates (lower is better); bar 3 is the share of answers replaced by the template, not an error rate.", "items": items}]


CAUTIONS = [
    "The audit (30 + 15 answers), the 50 test questions and the 60 new questions are small samples: read their rates as indicative, not precise.",
    "The checker and the group rules are heuristics. A pass is 'not flagged', not 'proven faithful'.",
    "Two thirds of evidence answers fall back to the rule-based template; the model's text is shown for the rest.",
    "Off-topic questions that contain a team name or the word 'shots' still get evidence (scope detector limitation).",
    "Only the five attributes are searchable: no player, minute or build-up questions.",
]


@lru_cache(maxsize=1)
def load_evaluation() -> dict:
    sections = _xg() + _retrieval() + _generation() + _faithfulness()
    return {"sections": sections, "cautions": CAUTIONS}
