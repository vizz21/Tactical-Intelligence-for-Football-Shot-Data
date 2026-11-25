#!/usr/bin/env python
"""Phase 5, step 2 tooling: show evidence blocks, load hand-written target batches through the grounding checker, finalise.

    uv run python scripts/phase5_targets.py show   train 0 40          # compact view of examples 0..39 (for the writer)
    uv run python scripts/phase5_targets.py add    train batch_01.txt  # lines: "<id> ::: <answer>"; runs the checker, prints failures
    uv run python scripts/phase5_targets.py add    train fix_01.txt --attempt 2    # the single regeneration of failures
    uv run python scripts/phase5_targets.py status train               # counts, first-pass rate, still-missing ids
    uv run python scripts/phase5_targets.py rescore train              # re-check every stored answer with the current checker
    uv run python scripts/phase5_targets.py finalize train             # -> data/processed/gen_targets_train.jsonl


"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from football_rag.grounding import check_answer          # noqa: E402
from football_rag.retrieval import is_relevant            # noqa: E402

DATA = ROOT / "data" / "processed"
WORK = DATA / "gen_targets_work"
SOURCE = "AI-generated: written by Claude (claude-sonnet-5-5) from the evidence block, kept only if the grounding checker passes"


def load(split):
    return [json.loads(l) for l in open(DATA / f"gen_inputs_{split}.jsonl", encoding="utf-8")]


def records(split):
    path = WORK / f"answers_{split}.jsonl"
    return [json.loads(l) for l in open(path, encoding="utf-8")] if path.exists() else []


def show(split, start, n):
    for ex in load(split)[start:start + n]:
        print(f"{ex['id']} [{ex['mode']}/{ex['kind']}] Q: {ex['query']} | status: {ex['retrieval']['status']}")
        for i, e in enumerate(ex["evidence"], 1):
            a = e["doc_attrs"]
            miss = ""
            if ex["query_attrs"]:
                bad = [f"{k}(asked {v})" for k, v in ex["query_attrs"].items() if a.get(k) != v]
                miss = f" !! mismatch: {', '.join(bad)}" if bad else ""
            print(f"  [{i}] {e['team_model']} m{e['minute']} | {e['player_model']} | {a['body']}, {a['side']}, {a['area']}, {a['yards']}yd"
                  f" | {e['outcome']} | xG {e['xg']:.2f}{miss}")
        if ex["evidence"]:
            print(f"  mean xG {ex['summary']['mean_xg']:.2f}, goals {ex['summary']['n_goals']}/{len(ex['evidence'])}")


def add(split, batch_file, attempt):
    by_id = {e["id"]: e for e in load(split)}
    WORK.mkdir(parents=True, exist_ok=True)
    prev = records(split)
    done_pass = {r["id"] for r in prev if r["passed"]}
    tried = {}
    for r in prev:
        tried[r["id"]] = max(tried.get(r["id"], 0), r["attempt"])
    n_ok, fails = 0, []
    out = open(WORK / f"answers_{split}.jsonl", "a", encoding="utf-8", newline="\n")
    for line in open(batch_file, encoding="utf-8"):
        line = line.strip()
        if not line or " ::: " not in line:
            continue
        ex_id, answer = line.split(" ::: ", 1)
        if ex_id not in by_id:
            print("unknown id", ex_id)
            continue
        if ex_id in done_pass:
            print("already accepted, skipped", ex_id)
            continue
        if tried.get(ex_id, 0) >= attempt:
            print(f"attempt {attempt} already recorded for", ex_id)
            continue
        res = check_answer(by_id[ex_id], answer)
        out.write(json.dumps({"id": ex_id, "attempt": attempt, "answer": answer, "passed": res["passed"], "reasons": res["reasons"]},
                             ensure_ascii=False) + "\n")
        if res["passed"]:
            n_ok += 1
        else:
            fails.append((ex_id, res["reasons"], answer))
    out.close()
    print(f"batch: {n_ok} passed, {len(fails)} failed")
    for ex_id, reasons, answer in fails:
        print(f"FAIL {ex_id}: {'; '.join(reasons)}\n     {answer}")


def rescore(split):
    """Re-run the CURRENT checker on every stored answer (after a checker fix). Keeps the old verdict in `passed_orig`."""
    by_id = {e["id"]: e for e in load(split)}
    recs = records(split)
    changed = 0
    for r in recs:
        r.setdefault("passed_orig", r["passed"])
        res = check_answer(by_id[r["id"]], r["answer"])
        changed += res["passed"] != r["passed"]
        r["passed"], r["reasons"] = res["passed"], res["reasons"]
    with open(WORK / f"answers_{split}.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    orig = [r for r in recs if r["attempt"] == 1]
    print(f"rescored {len(recs)} answers, verdict changed for {changed}; first-pass pass rate with the ORIGINAL checker "
          f"{sum(r['passed_orig'] for r in orig)}/{len(orig)}, with the current checker {sum(r['passed'] for r in orig)}/{len(orig)}")
    ok_ids = {r["id"] for r in recs if r["passed"]}
    for r in recs:
        if not r["passed"] and r["id"] not in ok_ids:
            print("NO PASSING ATTEMPT", r["id"], r["reasons"], "\n    ", r["answer"])


def status(split):
    inputs = load(split)
    recs = records(split)
    first = {}
    for r in recs:
        if r["attempt"] == 1:
            first[r["id"]] = r["passed"]
    ok = {r["id"] for r in recs if r["passed"]}
    written = {r["id"] for r in recs}
    pending_fix = sorted(i for i in first if not first[i] and i not in ok)
    print(f"{split}: inputs {len(inputs)} | written {len(written)} | first-pass pass rate "
          f"{sum(first.values())}/{len(first)} = {sum(first.values()) / max(1, len(first)):.3f} | accepted {len(ok)} | failed and not yet fixed {len(pending_fix)}")
    if pending_fix:
        print("failed, needs the single regeneration:", " ".join(pending_fix))
    missing = [e["id"] for e in inputs if e["id"] not in written]
    print(f"not written yet: {len(missing)}", (missing[:5], "...") if missing else "")


def finalize(split):
    inputs = {e["id"]: e for e in load(split)}
    recs = records(split)
    first, final, attempts = {}, {}, {}
    for r in recs:
        attempts[r["id"]] = max(attempts.get(r["id"], 0), r["attempt"])
        if r["attempt"] == 1:
            first[r["id"]] = r["passed"]
        if r["passed"]:
            final[r["id"]] = r
    rows = []
    for ex_id, ex in inputs.items():                      # keep input order
        if ex_id in final:
            r = final[ex_id]
            rows.append({"id": ex_id, "mode": ex["mode"], "kind": ex["kind"], "query": ex["query"], "input": ex["input"],
                         "target": r["answer"], "attempt": r["attempt"], "shot_ids": [e["shot_id"] for e in ex["evidence"]],
                         "target_source": SOURCE, "grounding_checked": True})
    with open(DATA / f"gen_targets_{split}.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"split": split, "n_inputs": len(inputs), "n_written": len(attempts), "first_pass_passed": sum(first.values()),
             "first_pass_rate": sum(first.values()) / max(1, len(first)), "passed_after_regeneration": sum(1 for i in final if not first.get(i)),
             "dropped_after_regeneration": sum(1 for i in attempts if i not in final), "n_final": len(rows),
             "by_mode": {m: sum(r["mode"] == m for r in rows) for m in ("answer", "partial", "refuse")}, "target_source": SOURCE}
    (ROOT / "results" / "phase5").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "phase5" / f"gen_targets_{split}_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    cmd, split = sys.argv[1], sys.argv[2]
    if cmd == "show":
        show(split, int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "add":
        att = int(sys.argv[sys.argv.index("--attempt") + 1]) if "--attempt" in sys.argv else 1
        add(split, sys.argv[3], att)
    elif cmd == "rescore":
        rescore(split)
    elif cmd == "status":
        status(split)
    elif cmd == "finalize":
        finalize(split)
