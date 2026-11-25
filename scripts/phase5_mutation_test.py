#!/usr/bin/env python
"""Phase 5: how well does the grounding checker catch planted errors?  (sanity check of the checker, no model involved)

    uv run python scripts/phase5_mutation_test.py val

Takes every ACCEPTED target of a split, corrupts it in one way at a time, and reports how often the checker rejects the
corrupted answer (detection rate) per mutation type. A mutation is only applied where the answer contains the thing to
corrupt. Writes results/phase5/grounding_mutation_{split}.json.
"""
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from football_rag.grounding import check_answer          # noqa: E402

DATA = ROOT / "data" / "processed"
OTHER_TEAMS = ["Real Madrid", "Bayern Munich", "Liverpool", "Juventus", "Argentina"]
OUTCOME_SWAP = [("went off target", "was blocked"), ("was saved", "was blocked"), ("was blocked", "was saved"), ("saved", "blocked"),
                ("blocked", "saved"), ("were goals", "were saved"), ("ended in goals", "were blocked")]


def mutations(ans, ex, rng):
    out = {}
    m = list(re.finditer(r"(?<![\w.])(\d\.\d\d)(?!\d)", ans))
    if m:
        k = rng.choice(m)
        v = float(k.group(1))
        out["xg_changed"] = ans[:k.start()] + f"{(v + 0.07 if v < 0.9 else v - 0.07):.2f}" + ans[k.end():]
    for a, b in OUTCOME_SWAP:
        if a in ans:
            out["outcome_swapped"] = ans.replace(a, b, 1)
            break
    m = list(re.finditer(r"\bminutes? (\d{1,3})\b", ans))
    if m and ex["evidence"]:
        k = rng.choice(m)
        n = int(k.group(1))
        out["minute_changed"] = ans[:k.start(1)] + str((n + 7) % 100 if (n + 7) % 100 not in {e["minute"] for e in ex["evidence"]} else (n + 11) % 100) + ans[k.end(1):]
    m = list(re.finditer(r"\b(\d{1,2}) yards\b", ans))
    if m:
        k = rng.choice(m)
        out["yards_changed"] = ans[:k.start(1)] + str(int(k.group(1)) + 9) + ans[k.end(1):]
    names = [e["player_model"] for e in ex["evidence"] if e["player_model"] and e["player_model"] in ans]
    if names:
        out["player_replaced"] = ans.replace(names[0], "Harry Kane", 1)
    teams = [e["team_model"] for e in ex["evidence"] if e["team_model"] in ans]
    if teams:
        out["team_replaced"] = ans.replace(teams[0], rng.choice(OTHER_TEAMS), 1)
    if ex["evidence"]:
        out["citation_out_of_range"] = re.sub(r"\[1\]", "[7]", ans, count=1) if "[1]" in ans else ans + " See [7]."
    return out


def main(split):
    rng = random.Random(0)
    inputs = {json.loads(l)["id"]: json.loads(l) for l in open(DATA / f"gen_inputs_{split}.jsonl", encoding="utf-8")}
    targets = [json.loads(l) for l in open(DATA / f"gen_targets_{split}.jsonl", encoding="utf-8")]
    stats, misses = {}, {}
    n_ok = 0
    for t in targets:
        ex = inputs[t["id"]]
        if ex["mode"] == "refuse":
            continue
        assert check_answer(ex, t["target"])["passed"]
        n_ok += 1
        for kind, bad in mutations(t["target"], ex, rng).items():
            s = stats.setdefault(kind, [0, 0])
            s[0] += 1
            caught = not check_answer(ex, bad)["passed"]
            s[1] += caught
            if not caught:
                misses.setdefault(kind, []).append((t["id"], bad))
    res = {"split": split, "n_answers_with_evidence": n_ok,
           "detection": {k: {"n": v[0], "caught": v[1], "rate": round(v[1] / v[0], 3)} for k, v in stats.items()},
           "examples_not_caught": {k: v[:3] for k, v in misses.items()}}
    (ROOT / "results" / "phase5").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "phase5" / f"grounding_mutation_{split}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res["detection"], indent=1))
    for k, v in misses.items():
        print("NOT CAUGHT", k, v[0])


if __name__ == "__main__":
    main(sys.argv[1])
