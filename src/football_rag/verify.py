"""Verify-and-fall-back for the generator (Phase 6).

The Phase 5 checker (`grounding.check_answer`) verifies per-item facts but not group claims (ranking, "all three ...",
counts) or which player/team a citation refers to; the Phase 6 audit found these were the main undetected errors.
`group_errors` adds simple text rules for them and `choose_answer` returns the model answer only if BOTH the Phase 5
checker and these rules pass; otherwise it returns the deterministic template answer (`template_answer`, arm D), which
is built from the evidence block and has no free-form claims.

The rules are written from the Phase 5 eval audit (development data) and then frozen; they are heuristics, not proofs.
Example: `ex` is a row of gen_inputs_*.jsonl (needs evidence, summary, retrieval).
"""
import re

from football_rag.gen_format import fold_text
from football_rag.grounding import check_answer, split_sentences

GOODISH = {"off target", "wayward", "hit the post"}
NEG = re.compile(r"\b(none|no goals?|not goals?|not a goal|zero)\b", re.I)
NUM = r"(\d\.\d{1,2})"
PART = r"(?:(?:de|da|do|dos|das|van|von|der|di|del|la|le|el|al|bin|ibn|ter|ten|mac|mc) )*"
CAP = r"[A-Z][\w'.-]*"
NAME = rf"({CAP}(?: {PART}{CAP}){{0,5}})"
BY = re.compile(rf"\[(\d)\] by {NAME}(?: of ([A-Z][\w' .-]*?))?(?= has | was | went | hit | took | scored | at | from |,|\(|\.| and | with |$)")
POSS = re.compile(rf"({CAP}(?: {PART}{CAP}){{1,5}}) (?:scored )?\[(\d)\]")
COUNT = {"one": 1, "two": 2, "three": 3}


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", re.sub(r"'s\b", "", fold_text(s).lower())))


def template_answer(ex: dict) -> str:
    """Arm D: rule-based answer from the evidence block (see scripts/phase5_report.py for the original definition)."""
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


def group_errors(ex: dict, answer: str) -> list[str]:
    """Names of the group-claim / attribution / garbling rules the answer breaks (empty list = none found)."""
    ev = ex["evidence"]
    if len(ev) < 3:
        return []
    answer = fold_text(answer)
    errs = set()
    max_xg = round(max(e["xg"] for e in ev), 2)
    n_goals = sum(e["goal"] for e in ev)
    for sent in split_sentences(answer):
        s = sent.lower()
        if "highest" in s:
            tail = s[s.index("highest"):]
            if "followed by" in tail:
                vals = [float(v) for v in re.findall(NUM, tail) if float(v) < 1.0]
                if len(vals) >= 2 and any(vals[i] < vals[i + 1] - 1e-9 for i in range(len(vals) - 1)):
                    errs.add("ordering")
            m = re.search(NUM, tail)
            if m and abs(float(m.group(1)) - max_xg) > 0.0051:
                errs.add("highest_value")
        if re.search(r"\ball\b", s) and not NEG.search(s):
            m = re.search(r"\ball [^.]*?\b(goals?|saved|blocked|off target|missed the target|went wide)\b", s)
            if m:
                w = m.group(1)
                ok = (lambda e: e["outcome"] in GOODISH) if w in ("off target", "missed the target", "went wide") else \
                     (lambda e, w=w: e["outcome"] == ("goal" if w.startswith("goal") else w))
                if not all(ok(e) for e in ev):
                    errs.add("all_outcome")
            for foot, key in (("right-footed", "right foot"), ("left-footed", "left foot")):
                if re.search(rf"\ball [^.]*?{foot}", s) and any(e["doc_attrs"]["body"] != key for e in ev):
                    errs.add("all_foot")
            for area in ("inside the box", "outside the box"):
                if re.search(rf"\ball [^.]*?\b{area}", s) and any(e["doc_attrs"]["area"] != area for e in ev):
                    errs.add("all_area")
            for word, key in (("central", "central"), ("left side", "left"), ("right side", "right")):
                if re.search(rf"\ball [^.]*?\b{word}", s) and any(e["doc_attrs"]["side"] != key for e in ev):
                    errs.add("all_side")
        else:
            claim = None
            if re.search(r"\b(none|no goals?)\b", s) and re.search(r"\bgoals?\b", s):
                claim = 0
            else:
                m = re.search(r"\b(one|two|three)(?: of [^.\[]{0,50}?)?(?: (?:was|were|is|are))? (?:a |the )?goals?\b", s)
                if m:
                    claim = COUNT[m.group(1)]
            if claim is not None and claim != n_goals:
                errs.add("goal_count")
        if len(s.split()) <= 4 and re.search(r"\[\d\]", s):
            errs.add("garbled")
    if re.search(r"\[\d\.\]|\]\s*\[|\bresults? \[\d\]\.", answer, re.I):
        errs.add("garbled")
    for m in BY.finditer(answer):
        i = int(m.group(1))
        if 1 <= i <= len(ev) and ev[i - 1].get("player_model"):
            if not _tokens(m.group(2)) <= _tokens(ev[i - 1]["player_model"]):
                errs.add("attribution_player")
            if m.group(3) and not _tokens(m.group(3)) <= _tokens(ev[i - 1]["team_model"]) | {"of"}:
                errs.add("attribution_team")
    for m in POSS.finditer(answer):
        i = int(m.group(2))
        if 1 <= i <= len(ev) and ev[i - 1].get("player_model") and not _tokens(m.group(1)) <= _tokens(ev[i - 1]["player_model"]):
            errs.add("attribution_player")
    return sorted(errs)


def choose_answer(ex: dict, model_answer: str) -> tuple[str, str, list[str]]:
    """(text, source, reasons): the model answer if it passes the Phase 5 checker and the group rules, else the template."""
    res = check_answer(ex, model_answer)
    reasons = list(res["reasons"]) if not res["passed"] else []
    reasons += [f"group:{e}" for e in group_errors(ex, model_answer)]
    if reasons:
        return template_answer(ex), "template", reasons
    return fold_text(model_answer), "model", []
