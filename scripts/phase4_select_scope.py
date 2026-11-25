#!/usr/bin/env python
"""Phase 4 close-out, task 1: choose the low-confidence / scope rule. VALIDATION queries only.

    uv run python scripts/phase4_select_scope.py

SELECTION RULE (written before running):
  options : OLD  single threshold 0.620 (comparison only, not eligible)
            (a)  threshold per number of recognised attributes (0, 1, 2, 3+); each = 5th percentile of the
                 in-scope CALIBRATION queries' top-1 similarity in that group
            (b)  query contains a known attribute word or shot-domain word (keyword lexicon, scope.py) ->
                 in scope whatever the similarity; otherwise similarity >= 0.620 is the backstop
  data    : calibration half (fits (a)) and a held-out EVALUATION half; all queries are validation-side
            (template queries from val shots/teams, hand-written short/off-topic sentences). Test is not used.
  score   : combined = mean false-reject rate over 5 equally weighted in-scope groups
            (template 1 / 2 / 3 attributes, short natural, long paraphrase = para_dev)
            + false-accept rate over a pool of off-topic, hard off-topic and minute/player queries.
            false-accept: off-topic query treated as in scope; minute/player query served with no
            unsupported note or with evidence although no attribute was recognised.
  choice  : lowest combined of (a) and (b); tie -> (b) (fewer fitted parameters).
"""
import ast
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from football_rag import config as C                       # noqa: E402
from football_rag import retrieval as R                    # noqa: E402
from football_rag.pipeline import Pipeline                 # noqa: E402
from football_rag.scope import decide, group_of            # noqa: E402

OLD_THR = json.loads((C.RESULTS_P4 / "retrieval_check.json").read_text(encoding="utf-8"))["low_confidence_threshold"]

SHORT_CAL = ["shots", "attempts", "goals", "headers", "saved shots", "blocked shots", "missed shots", "left foot shots",
             "right foot shots", "shots from the left", "shots from the right", "shots from the centre", "shots inside the box",
             "shots outside the box", "long shots", "headed goals", "saved headers", "close range shots"]
SHORT_EVAL = ["efforts", "chances", "header goals", "blocked headers", "left-footed shots", "right footed goals",
              "goals from distance", "centre shots", "wide shots", "missed headers", "saved efforts", "shots from the left wing"]
OFF_CAL = ["what is the weather in london today", "how do I bake sourdough bread", "best laptop for machine learning",
           "translate good morning into french", "who is the president of france", "how many people live in tokyo",
           "recommend a movie for tonight", "what time does the train leave", "explain how photosynthesis works",
           "stock market news this week", "how to learn the guitar quickly", "directions to the nearest pharmacy",
           "tell me a joke", "what is the capital of australia", "how do airplanes stay in the air"]
OFF_EVAL = ["what is the weather in Paris tomorrow", "who won the world cup in 2010", "recipe for chocolate cake",   # given
            "how tall is mount everest", "what should I cook for dinner", "who painted the mona lisa", "best way to lose weight",
            "when does the next semester start", "how to reset my phone password", "latest news about the election",
            "what is the population of canada", "play some relaxing music", "how does a car engine work"]
OFF_HARD = ["what is the best goal in life", "how do I save a file on my computer", "left or right wing politics",
            "shots of espresso at the cafe", "blocked number on my phone", "head of the department email"]
UNSUP_EVAL = ["free kicks taken in minute 85", "all shots by Lionel Messi", "Messi left foot goals",                  # given
              "goals in the 85th minute", "shots by Cristiano Ronaldo", "Neymar right foot goals",
              "saved shots in the first half", "late goals after the 80' mark"]
NAMED = ["shots", "headers", "shots by Barcelona"]
GIVEN_5 = ["what is the weather in Paris tomorrow", "who won the world cup in 2010", "recipe for chocolate cake",
           "free kicks taken in minute 85", "all shots by Lionel Messi"]

PH = {"body": ["with the left foot", "with the right foot", "with the head"],
      "side": ["from the left side", "from the right side", "from a central position"],
      "area": ["from inside the box", "from outside the box"],
      "outcome": ["that was saved", "that ended in a goal", "that was blocked", "that missed the target"]}


def para_dev():
    nb = json.load(open(C.ROOT / "notebooks" / "03_biencoder_clean.ipynb", encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if src.lstrip().startswith("PARA = ["):
            for node in ast.parse(src).body:
                if isinstance(node, ast.Assign) and node.targets[0].id == "PARA":
                    return [q for q, _ in ast.literal_eval(node.value)[0::2]]


def pct(x):
    x = np.asarray(x)
    return {"n": len(x), "min": x.min(), "p5": np.percentile(x, 5), "p25": np.percentile(x, 25),
            "median": np.median(x), "p95": np.percentile(x, 95), "max": x.max()}


def main():
    rng = random.Random(0)
    p = Pipeline(rule={"name": "old", "backstop": OLD_THR})
    tab = p.tab
    val_teams = sorted(tab.loc[tab["split"] == "val", "team"].unique())

    # ---- query sets: (text, group, half)
    Q = []
    seen = {}
    for line in open(C.PAIRS, encoding="utf-8"):
        d = json.loads(line)
        if d["split"] == "val":
            seen.setdefault(d["query"], len(d["attrs"]))
    items = sorted(seen.items())
    rng.shuffle(items)
    for i, (q, n) in enumerate(items):
        Q.append((q, f"tmpl{n}", "cal" if i % 2 == 0 else "eval"))
    one = [f"Find a shot by {t}" for t in val_teams] + [f"Find a shot {ph}" for v in PH.values() for ph in v]
    rng.shuffle(one)
    Q += [(q, "tmpl1", "cal" if i % 2 == 0 else "eval") for i, q in enumerate(one)]
    Q += [(q, "short", "cal") for q in SHORT_CAL] + [(f"shots by {t}", "short", "cal") for t in val_teams[0::2]]
    Q += [(q, "short", "eval") for q in SHORT_EVAL]
    Q += [(f"{t} shots" if i % 2 == 0 else f"goals by {t}", "short", "eval") for i, t in enumerate(val_teams[1::2])]
    Q += [(q, "long", "eval") for q in para_dev()]
    Q += [(q, "off", "cal") for q in OFF_CAL] + [(q, "off", "eval") for q in OFF_EVAL]
    Q += [(q, "off_hard", "eval") for q in OFF_HARD] + [(q, "unsupported", "eval") for q in UNSUP_EVAL]
    Q += [(q, "named", "eval") for q in NAMED]
    df = pd.DataFrame(Q, columns=["query", "group", "half"])

    emb = R.embed(p.encoder, df["query"].tolist())
    df["top1"] = p.index.similarities(emb).max(1)
    parsed = [p.scope.parse(q) for q in df["query"]]
    df["n_attrs"] = [x.n_attrs for x in parsed]
    notes = [p.detect_unsupported(q) for q in df["query"]]
    df["has_note"] = [bool(n) for n in notes]

    # ---- similarity percentiles per group (all queries of the group)
    prc = pd.DataFrame({g: pct(d["top1"]) for g, d in df.groupby("group")}).T
    print("top-1 similarity percentiles per group (calibration + evaluation halves)")
    print(prc.round(3).to_string())

    # ---- fit (a) on calibration in-scope queries
    cal = df[(df["half"] == "cal") & df["group"].isin(["tmpl1", "tmpl2", "tmpl3", "short"])]
    gt = {}
    for g in ["0", "1", "2", "3+"]:
        v = cal.loc[[group_of(n) == g for n in cal["n_attrs"]], "top1"]
        gt[g] = float(np.percentile(v, 5)) if len(v) else OLD_THR
        print(f"(a) group {g}: n={len(v)} calibration queries, threshold {gt[g]:.3f}")
    rules = {"old": {"name": "old", "backstop": OLD_THR},
             "(a) per attribute count": {"name": "per_attr_count", "backstop": OLD_THR, "group_thresholds": gt},
             "(b) lexicon + backstop": {"name": "lexicon_backstop", "backstop": OLD_THR}}

    # ---- evaluate on the held-out half
    ev = df[df["half"] == "eval"].copy()
    rows, items_out = [], []
    for rname, rule in rules.items():
        res = []
        for (_, r), ps, nt in zip(ev.iterrows(), [parsed[i] for i in ev.index], [notes[i] for i in ev.index]):
            low, withheld, _ = decide(rule, ps, r["top1"], nt)
            if r["group"] in ("tmpl1", "tmpl2", "tmpl3", "short", "long", "named"):
                bad = low                                           # false reject
            elif r["group"] in ("off", "off_hard"):
                bad = not low                                       # false accept
            else:                                                   # unsupported: served as if understood
                bad = (not nt) or ((not withheld) and ps.n_attrs == 0)
            res.append(bad)
            items_out.append({"rule": rname, "query": r["query"], "group": r["group"], "top1": round(r["top1"], 3),
                              "n_attrs": r["n_attrs"], "low_confidence": low, "withheld": withheld, "error": bad})
        ev["bad"] = res
        g = ev.groupby("group")["bad"].mean()
        frr = [g[k] for k in ("tmpl1", "tmpl2", "tmpl3", "short", "long")]
        pool = ev[ev["group"].isin(["off", "off_hard", "unsupported"])]
        far = pool["bad"].mean()
        rows.append({"rule": rname, "FRR tmpl1": g["tmpl1"], "FRR tmpl2": g["tmpl2"], "FRR tmpl3": g["tmpl3"],
                     "FRR short": g["short"], "FRR long": g["long"], "mean FRR": np.mean(frr),
                     "FAR off-topic": g["off"], "FAR hard off-topic": g["off_hard"], "FAR minute/player": g["unsupported"],
                     "pooled FAR": far, "combined": np.mean(frr) + far, "FRR named (3)": g["named"]})
    table = pd.DataFrame(rows)
    counts = ev.groupby("group").size().to_dict()
    print("\nheld-out evaluation counts:", counts)
    print(table.set_index("rule").T.round(3).to_string())

    cand = table[table["rule"] != "old"].set_index("rule")["combined"]
    best = "(b) lexicon + backstop" if cand["(b) lexicon + backstop"] <= cand["(a) per attribute count"] else "(a) per attribute count"
    print(f"\nCHOSEN: {best} (combined {cand[best]:.3f}; old rule {table.set_index('rule').loc['old', 'combined']:.3f})")

    # given 5 off-topic examples and the 3 named queries under each rule
    it = pd.DataFrame(items_out)
    print("\nper-query view of the user-named examples")
    show = it[it["query"].isin(GIVEN_5 + NAMED + ["Messi left foot goals"])]
    print(show.pivot_table(index=["query", "top1"], columns="rule", values="low_confidence", aggfunc="first").to_string())

    C.RESULTS_P4.mkdir(parents=True, exist_ok=True)
    table.to_csv(C.RESULTS_P4 / "scope_selection.csv", index=False)
    it.to_csv(C.RESULTS_P4 / "scope_selection_items.csv", index=False)
    prc.to_csv(C.RESULTS_P4 / "scope_top1_percentiles.csv")
    out = {**rules[best], "chosen": best, "eval_counts": counts, "group_thresholds_fitted": gt}
    (C.RESULTS_P4 / "scope_rule.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {C.RESULTS_P4 / 'scope_rule.json'}")


if __name__ == "__main__":
    main()
