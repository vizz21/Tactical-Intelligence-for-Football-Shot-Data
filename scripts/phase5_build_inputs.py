#!/usr/bin/env python
"""Phase 5, step 1: build the generator inputs from the REAL pipeline (retrieval + xG). No training here.

    uv run python scripts/phase5_build_inputs.py

Writes data/processed/gen_inputs_{train,val,eval}.jsonl, data/processed/gen_inputs_eval.sha256 and
results/phase5/gen_inputs_stats.json.

  train : 1,000 queries, evidence from TRAIN-split shots only  (450 template + 450 phrase-bank + 100 off-topic/unsupported)
  val   :   100 queries, evidence from VAL-split shots only    ( 45 template +  45 phrase-bank +  10 off-topic/unsupported)
  eval  :   143 queries, evidence from TEST-split shots only   (100 template + 28 fresh near/far + 15 off-topic/unsupported)
          FIXED: never used for training or selection. Written once; its SHA-256 is recorded.

Query texts are disjoint across the three files. Out-of-scope queries (the pipeline's scope check) get NO evidence in
the model input, only the status line, so the model learns to say it cannot answer.
"""
import ast
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from football_rag import config as C                                    # noqa: E402
from football_rag.gen_format import evidence_for_model, fold_text, format_input, outcome_word, retrieval_status   # noqa: E402
from football_rag.pipeline import Pipeline                              # noqa: E402
from football_rag.retrieval import is_relevant                          # noqa: E402

OUT = C.DATA
STATS = C.ROOT / "results" / "phase5"
NB = C.ROOT / "notebooks" / "03_biencoder_clean.ipynb"

OFF_TRAIN = ["what is the weather like in London today", "how do I cook pasta carbonara", "best way to learn python quickly",
             "who is the prime minister of canada", "translate hello into spanish", "how many calories are in an apple",
             "recommend a good book about history", "what time is it in Tokyo", "tell me a funny story", "how do I fix a flat bicycle tire",
             "what is the tallest mountain in the world", "can you write me a poem about the sea", "how does a vaccine work",
             "plan a three day trip to Rome", "what is the exchange rate for dollars to euros", "explain how electricity works",
             "what should I name my puppy", "how to start a vegetable garden", "who painted starry night", "what are the symptoms of a cold",
             "best movies of the last decade", "how do I bake chocolate chip cookies", "what is the speed of light",
             "give me tips for a job interview", "how far is the moon from earth", "what is machine learning", "how do I change a lightbulb",
             "which phone should I buy", "what is the meaning of life", "how do airplanes fly", "who wrote hamlet",
             "what is the population of brazil", "help me with my math homework", "how do I lose weight fast", "what is the stock price of apple",
             "tell me about the roman empire", "how to make a paper airplane", "why is the sky blue", "explain the rules of chess",
             "how do I make coffee at home", "what are good names for a cat", "how does the internet work", "what is the longest river in africa",
             "how do I write a cover letter", "what is the best programming language", "how do I get better sleep", "recommend a podcast about science", "how do I plant tomatoes",
             "what is a good workout routine", "explain how a battery works", "how do I tie a tie", "what is the history of jazz", "how do I clean a laptop screen",
             "what is the boiling point of water", "how do I learn to swim", "what is the capital of peru"]
OFF_VAL = ["what is the weather forecast for Berlin", "how do I bake sourdough bread", "recommend a good podcast", "what is the tallest building in the world",
           "how do I reset my phone"]
OFF_EVAL = ["what is the weather like in Paris tomorrow", "recipe for banana bread", "how do I reset my router", "how tall is the Eiffel tower",
            "tell me a joke about cats"]
MIN_ONLY = ["shots in the {n}th minute", "what happened in minute {n}", "{n}th minute efforts", "attempts after minute {n}",
            "chances in the first half", "efforts in the second half", "every shot before the {n}th minute", "anything in the last ten minutes",
            "shots around minute {n}", "late attempts after the {n}' mark", "strikes in the opening minutes"]
PLAYER_ONLY = ["shots by {p}", "attempts from {p}", "show me {p}'s efforts", "{p} chances", "every effort by {p}", "what shots did {p} take",
               "all strikes by {p}", "{p} attempts at goal"]
PARTIAL_PLAYER = ["{p} left foot goals", "{p} headers", "saved shots by {p}", "{p} shots from outside the box", "goals by {p} from the right side",
                  "{p} right foot shots that were blocked"]
PARTIAL_MIN = ["goals in the {n}th minute", "saved headers after minute {n}", "left foot goals in the first half", "blocked shots in minute {n}",
               "shots from the right side in the {n}th minute", "headed goals in the second half", "right foot shots before minute {n}"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def load_notebook_defs():
    """BANK, TEAM_FORMS, OPENERS, make_aug_queries, ATTRS, FRESH_NEAR/FAR read from notebook 03 (no copies)."""
    nb = json.load(open(NB, encoding="utf-8"))
    ns = {"random": random}
    want = {"ATTRS", "BANK", "TEAM_FORMS", "OPENERS"}
    fresh = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in want:
                    exec(compile(ast.Module([node], []), "nb", "exec"), ns)
                if name in ("FRESH_NEAR", "FRESH_FAR"):
                    try:
                        v = ast.literal_eval(node.value)
                        if len(v) > 1:
                            fresh[name] = v
                    except ValueError:
                        pass
            elif isinstance(node, ast.FunctionDef) and node.name == "make_aug_queries":
                exec(compile(ast.Module([node], []), "nb", "exec"), ns)
    return ns, fresh


class Builder:
    def __init__(self):
        self.p = Pipeline(device="cpu")
        self.tab = self.p.tab
        self.corpus = self.p.corpus
        self.ns, self.fresh = load_notebook_defs()
        self.pairs = [json.loads(l) for l in open(C.PAIRS, encoding="utf-8")]
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained("google/flan-t5-base")

    # ---------------------------------------------------------------- one example
    def example(self, qid, pool, kind, query, attrs=None):
        res = self.p.search(query, 3, splits={pool})
        items = evidence_for_model(res)
        withheld = not res["evidence"]
        status = retrieval_status(res["summary"], withheld)
        notes = res["summary"]["unsupported"]
        mode = "refuse" if not items else ("partial" if notes else "answer")
        ev = []
        for e in items:
            row_idx = self.p.shot_pos[e["shot_id"]]
            assert outcome_word(e["document"]) and (e["goal"] == 1) == (outcome_word(e["document"]) == "goal")
            ev.append({"rank": e["rank"], "shot_id": e["shot_id"], "row_idx": row_idx, "match_id": e["match_id"], "team": e["team"],
                       "player": e["player"], "minute": e["minute"], "period": e["period"], "document": e["document"],
                       "doc_attrs": e["doc_attrs"], "xg": e["xg"], "outcome": outcome_word(e["document"]), "goal": e["goal"],
                       "split": e["split"], "score": e["score"], "document_model": fold_text(e["document"]),
                       "player_model": fold_text(e["player"]) if e["player"] else None, "team_model": fold_text(e["team"]),
                       "relevant": None if attrs is None else bool(is_relevant(attrs, e["doc_attrs"]))})
        n_rel = None
        if attrs is not None:
            mask = self.p.split_arr == pool
            n_rel = int(sum(is_relevant(attrs, self.corpus[i]["doc_attrs"]) for i in np.where(mask)[0]))
        return {"id": qid, "pool": pool, "kind": kind, "mode": mode, "query": query, "query_attrs": attrs,
                "input": format_input(query, items, status), "evidence": ev,
                "summary": {"n_results": len(ev), "mean_xg": float(np.mean([e["xg"] for e in ev])) if ev else None,
                            "n_goals": int(sum(e["goal"] for e in ev))},
                "retrieval": {"status": status, "low_confidence": res["summary"]["low_confidence"],
                              "unsupported": notes, "understood": res["summary"]["understood"]},
                "n_relevant_in_pool": n_rel}

    # ---------------------------------------------------------------- query sources
    def template_pool(self, split):
        d = {}
        for pr in self.pairs:
            if pr["split"] == split:
                d.setdefault(pr["query"], pr["attrs"])
        return sorted(d.items())

    def bank_queries(self, split, rng, n, n_single):
        ns = self.ns
        shots = [c for c in self.corpus if c["split"] == split]
        out = []
        for _ in range(n * 3):
            c = rng.choice(shots)
            if len(out) < n_single:
                key = rng.choice(ns["ATTRS"])
                frag = rng.choice(ns["TEAM_FORMS"]).format(c["doc_attrs"]["team"]) if key == "team" else rng.choice(ns["BANK"][key][c["doc_attrs"][key]])
                out.append((rng.choice(ns["OPENERS"]) + " " + frag, {key: c["doc_attrs"][key]}))
            else:
                q = ns["make_aug_queries"](c["doc_attrs"], rng, 1)[0]
                out.append((q["query"], q["attrs"]))
        return out

    def players(self, split):
        names = sorted(self.tab.loc[self.tab["split"] == split, "player"].dropna().unique())
        return names

    def player_ref(self, rng, names):
        full = rng.choice(names)
        toks = [t for t in full.split() if len(t) >= 5]
        return full if (rng.random() < 0.4 or not toks) else toks[-1]

    def unsupported_candidates(self, split, rng, n_min, n_player, n_partial):
        names, out = self.players(split), []
        num = lambda: rng.choice(list(range(6, 90)))
        for _ in range(n_min * 4):
            out.append(("minute", rng.choice(MIN_ONLY).format(n=num())))
        for _ in range(n_player * 4):
            out.append(("player", rng.choice(PLAYER_ONLY).format(p=self.player_ref(rng, names))))
        for _ in range(n_partial * 4):
            out.append(("partial", rng.choice(PARTIAL_PLAYER).format(p=self.player_ref(rng, names)) if rng.random() < 0.5
                        else rng.choice(PARTIAL_MIN).format(n=num())))
        return out

    # ---------------------------------------------------------------- assemble one file
    def build(self, pool, seed, n_template, n_bank, n_single, off_list, n_off, n_min, n_player, n_partial, used, fresh=False, n_fresh_template=None):
        rng = random.Random(seed)
        exs, texts = [], set()

        def add(kind, q, attrs, want_mode, counter, limit):
            if q in used or q in texts or counter.get(kind, 0) >= limit:
                return
            ex = self.example(f"{pool}-{len(exs):04d}", pool, kind, q, attrs)
            if want_mode is not None and ex["mode"] != want_mode:
                return
            if len(self.tok(ex["input"]).input_ids) > 512:               # keep every input within 512 tokens
                return
            exs.append(ex); texts.add(q); counter[kind] = counter.get(kind, 0) + 1

        cnt = {}
        tpl = self.template_pool(pool)
        rng.shuffle(tpl)
        for q, a in tpl:
            add("template", q, a, "answer", cnt, n_template)
        if n_bank:
            for q, a in self.bank_queries(pool, rng, n_bank + 200, n_single):
                add("bank", q, a, "answer", cnt, n_bank)
        if fresh:
            near, far = self.fresh["FRESH_NEAR"], self.fresh["FRESH_FAR"]
            for kind, items in (("fresh_near", near), ("fresh_far", far)):
                for q, a in items:
                    n_rel = sum(is_relevant(a, c["doc_attrs"]) for c in self.corpus if c["split"] == pool)
                    if n_rel == 0:
                        print(f"  dropped fresh query with no relevant {pool} document: {q!r}")
                        continue
                    add(kind, q, a, None, cnt, 99)         # kept whatever the scope check does (real system behaviour)
        off = list(off_list)
        rng.shuffle(off)
        for q in off:
            add("off_topic", q, None, "refuse", cnt, n_off)
        cands = self.unsupported_candidates(pool, rng, n_min, n_player, n_partial)
        for kind, q in cands:
            if kind == "partial":
                add("partial", q, None, "partial", cnt, n_partial)
            else:
                add(f"unsupported_{kind}", q, None, "refuse", cnt, n_min if kind == "minute" else n_player)
        return exs, texts, cnt


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    b = Builder()
    used = set()
    # eval first (fixed), then val, then train, so later sets exclude earlier query texts
    ev, t, c_ev = b.build("test", 2, 100, 0, 0, OFF_EVAL, 5, 3, 3, 4, used, fresh=True)
    used |= t
    va, t, c_va = b.build("val", 1, 45, 45, 5, OFF_VAL, 5, 2, 2, 1, used)
    used |= t
    tr, t, c_tr = b.build("train", 0, 450, 450, 45, OFF_TRAIN, 50, 17, 17, 16, used)
    for name, rows, cnt in (("eval", ev, c_ev), ("val", va, c_va), ("train", tr, c_tr)):
        print(name, len(rows), cnt)
    ids = [{e["query"] for e in r} for r in (tr, va, ev)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2]), "query texts overlap between files"
    for e in tr:
        assert all(x["split"] == "train" for x in e["evidence"])
    for e in va:
        assert all(x["split"] == "val" for x in e["evidence"])
    for e in ev:
        assert all(x["split"] == "test" for x in e["evidence"])
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "gen_inputs_train.jsonl", tr)
    write_jsonl(OUT / "gen_inputs_val.jsonl", va)
    write_jsonl(OUT / "gen_inputs_eval.jsonl", ev)
    sha = sha256(OUT / "gen_inputs_eval.jsonl")
    (OUT / "gen_inputs_eval.sha256").write_text(f"{sha}  gen_inputs_eval.jsonl\n", encoding="ascii")
    print("eval sha256", sha)

    # ---- token lengths with the flan-t5 tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    stats = {"eval_sha256": sha}
    for name, rows in (("train", tr), ("val", va), ("eval", ev)):
        enc = [tok(r["input"]).input_ids for r in rows]
        lens = np.array([len(x) for x in enc])
        n_unk = np.array([sum(i == tok.unk_token_id for i in x) for x in enc])
        modes = {m: sum(r["mode"] == m for r in rows) for m in ("answer", "partial", "refuse")}
        stats[name] = {"n": len(rows), "modes": modes, "kinds": {k: sum(r["kind"] == k for r in rows) for k in sorted({r["kind"] for r in rows})},
                       "tokens_mean": float(lens.mean()), "tokens_p95": float(np.percentile(lens, 95)), "tokens_max": int(lens.max()),
                       "n_over_512": int((lens > 512).sum()), "n_with_unk_tokens": int((n_unk > 0).sum()),
                       "mean_relevant_in_top3": float(np.mean([np.mean([x["relevant"] for x in r["evidence"]]) for r in rows if r["evidence"] and r["query_attrs"]]))}
        print(name, json.dumps(stats[name]))
    STATS.mkdir(parents=True, exist_ok=True)
    (STATS / "gen_inputs_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
