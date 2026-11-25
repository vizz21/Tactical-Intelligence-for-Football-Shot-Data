"""Grounding checker for generated answers (Phase 5). A HEURISTIC: regexes plus name lists from tabular.csv.

check_answer(example, answer) compares an answer with the evidence block the model was given and returns
  xg_ok / xg_details      (a) every decimal stated as xG matches the evidence xG (within rounding)
  outcome_ok / outcome_errors   (b) outcome words (goal / saved / blocked / off target) agree with the cited shot
  unsupported             (c) claims not in the evidence: player or other proper names, teams, minutes, yard
                              distances, body part, side, area
  citations_ok, format_ok, passed

Text is compared ASCII-folded (the model sees folded names, see gen_format.fold_text). What it does NOT check:
tactical or causal claims, counts ("two of the three"), the build-up passes, paraphrase quality, whether an answer is
useful. Alignment of a claim with a shot uses the nearest "[n]" citation (consecutive citations such as "[1] and [2]"
form a group and every shot in the group must satisfy the claim). A sentence with no citation is checked against the
union of all evidence. Expect false alarms and misses; use it to filter training targets and as one evaluation signal.
"""
import re
from functools import lru_cache

import pandas as pd

from .config import TABULAR
from .gen_format import fold_text

CITE = re.compile(r"\[(\d+)\]")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[\"'])")
DECIMAL = re.compile(r"(?<![\w.])(\d+\.\d+)(?!\d)")
PERCENT = re.compile(r"\d+(?:\.\d+)?\s?%")
NEGATION = re.compile(r"\b(?:no|none|not|neither|nor|never|without|zero)\b|n't|\b0 of\b", re.I)
ALLOWED_CAPS = {"xg", "i", "ok"}

# outcome category of the evidence outcome wording; answer terms per category
OUTCOME_CATS = {"goal": {"G"}, "saved": {"S"}, "blocked": {"B"}, "off target": {"M"}, "wayward": {"M"}, "hit the post": {"M"},
                "saved, off target": {"S", "M"}, "saved onto the post": {"S", "M"}}
OUTCOME_TERMS = {
    "G": r"\bgoals?\b|\bscored\b|\bwent in\b|\bfound the net\b|\bbeat the (?:keeper|goalkeeper)\b",
    "S": r"\bsaved?\b|\bsaves\b|\bparried\b|\bstopped by the (?:keeper|goalkeeper)\b",
    "B": r"\bblocked?\b|\bblocks?\b",
    "M": r"\boff target\b|\bwayward\b|\bhit the post\b|\bhitting the post\b|\bpost\b|\bwent wide\b|\bwide\b|\bmissed\b|\bmisses\b|\bover the bar\b",
}
OUTCOME_EXCLUDE = re.compile(r"expected goals?|goal probability|chance of (?:a )?(?:goal|scoring)|probability of (?:a )?goal|scoring chance|"
                             r"\bat goal\b|\bon goal\b|goal-?mouth|goal-?bound|goal-scoring", re.I)
BODY_TERMS = {"left foot": r"\bleft[- ]foot(?:ed)?\b|\bleft boot\b", "right foot": r"\bright[- ]foot(?:ed)?\b|\bright boot\b",
              "head": r"\bheaders?\b|\bheaded\b|\bheading\b|\bwith the head\b|\bnodded\b"}
SIDE_TERMS = {"left": r"\bleft\b", "right": r"\bright\b", "central": r"\bcent(?:ral|re|er)\b|\bmiddle\b"}
AREA_TERMS = {"inside the box": r"\b(?:inside|in|within) the (?:penalty |18-yard )?(?:box|area)\b|\bclose range\b|\bsix-yard\b",
              "outside the box": r"\boutside the (?:penalty |18-yard )?(?:box|area)\b|\bfrom distance\b|\blong[- ]range\b|\bfrom range\b"}
MINUTE_PATTERNS = [r"\bminutes?\s+(\d{1,3})\b", r"\b(\d{1,3})(?:st|nd|rd|th)\s+minute", r"\b(\d{1,3})'", r"\bmin\.?\s*(\d{1,3})\b",
                   r"\b(\d{1,3})\s*(?:minutes?|mins?)\b"]
YARDS = re.compile(r"\b(\d{1,3})\s*-?\s*(?:yards?|yds?)\b(?!\s*(?:box|line|area))|\b(\d{1,3})\s*-?\s*(?:metres?|meters?)\b", re.I)


# ------------------------------------------------------------------------------------------ vocabularies
@lru_cache(maxsize=1)
def vocab():
    tab = pd.read_csv(TABULAR)
    team_names = {fold_text(t).casefold() for t in tab["team"].unique()}
    teams = set(team_names)
    teams |= {t[: -len(" women's")] for t in team_names if t.endswith(" women's")}
    team_tokens = {w for t in team_names for w in re.findall(r"\w+", t)}
    names = {fold_text(p).casefold() for p in tab["player"].dropna().unique()}
    name_tokens = {w for n in names for w in re.findall(r"\w+", n) if len(w) >= 4} - team_tokens
    team_re = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(t) for t in sorted(teams, key=len, reverse=True)) + r")(?!\w)")
    return {"team_re": team_re, "name_tokens": name_tokens}


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold()))


# ------------------------------------------------------------------------------------------ helpers
def split_sentences(text: str) -> list[str]:
    return [s for s in SENT_SPLIT.split(text.strip()) if s]


def citation_groups(sent: str) -> list[dict]:
    """Consecutive citations ("[1] and [2]", "[1], [2]") form one group."""
    groups = []
    for m in CITE.finditer(sent):
        n = int(m.group(1))
        if groups and re.fullmatch(r"\s*(?:,|&|or|/)?\s*(?:and\s*)?", sent[groups[-1]["end"]:m.start()]):
            groups[-1]["end"] = m.end()
            groups[-1]["items"].append(n)
        else:
            groups.append({"start": m.start(), "end": m.end(), "items": [n]})
    return groups


DIRECT_OBJECT = re.compile(r"\s*(?:(?:the|a|an|that|this|those|these|shot|shots|effort|efforts|attempt|attempts|chance|one|ones)\s+)*", re.I)


def items_for(sent: str, start: int, end: int, groups: list[dict], n_ev: int):
    """Evidence indices (0-based) a claim at [start, end) of `sent` refers to, or None (sentence has no citation).
    A claim that stands directly before a citation ("scored [1]", "saved the shot [2]") belongs to that citation;
    otherwise it belongs to the most recent preceding citation ("[2], a left-footed effort ... in minute 53");
    a claim before every citation belongs to the first one after it."""
    if not groups:
        return None
    pre = [g for g in groups if g["end"] <= start]
    post = [g for g in groups if g["start"] >= end]
    best = None
    if post:
        nxt = min(post, key=lambda g: g["start"])
        if DIRECT_OBJECT.fullmatch(sent[end:nxt["start"]]):
            best = nxt
    if best is None and pre:
        best = max(pre, key=lambda g: g["end"])
    if best is None and post:
        best = min(post, key=lambda g: g["start"])
    return [i - 1 for i in best["items"] if 1 <= i <= n_ev]


def _negated(sent: str, start: int) -> bool:
    return bool(NEGATION.search(sent[max(0, start - 40):start]))


def evidence_cats(ev: dict) -> set[str]:
    return OUTCOME_CATS.get(ev["outcome"], set())


# ------------------------------------------------------------------------------------------ the checks
def check_xg(sentences, evidence, mean_xg):
    details, ok = [], True
    for sent in sentences:
        groups = citation_groups(sent)
        cited = sorted({i - 1 for g in groups for i in g["items"] if 1 <= i <= len(evidence)})
        pool = [evidence[i]["xg"] for i in cited] if cited else [e["xg"] for e in evidence]
        allowed = [(x, f"{x:.2f}") for x in pool] + ([(mean_xg, f"{mean_xg:.2f}")] if mean_xg is not None else [])
        for m in DECIMAL.finditer(sent):
            v = float(m.group(1))
            good = any(abs(v - x) <= 0.005 + 1e-9 or abs(v - float(d)) < 1e-9 for x, d in allowed)
            details.append({"stated": v, "ok": good, "sentence": sent})
            ok &= good
        if PERCENT.search(sent):
            details.append({"stated": PERCENT.search(sent).group(0), "ok": False, "sentence": sent})
            ok = False
    return ok, details


def _scan_terms(sent, terms: dict, evidence, value_of, kind, errors, question, label_cats=False):
    """Generic claim scan: a term of `kind` in the sentence must hold for the shot(s) it is aligned with.
    A term in a sentence with no citation that merely restates the question ("I searched only left-footed goals")
    is not a claim."""
    groups = citation_groups(sent)
    text = OUTCOME_EXCLUDE.sub(lambda m: " " * len(m.group(0)), sent) if label_cats else sent
    for key, pat in terms.items():
        for m in re.finditer(pat, text, re.I):
            if label_cats and _negated(text, m.start()):
                continue
            idx = items_for(sent, m.start(), m.end(), groups, len(evidence))
            if not idx and re.search(pat, question, re.I):
                continue
            pool = [evidence[i] for i in idx] if idx else evidence
            if not pool:
                continue
            holds = [key in value_of(e) for e in pool]
            good = all(holds) if idx else any(holds)
            if not good:
                errors.append({"type": kind, "text": m.group(0), "claim": key, "sentence": sent})


def check_attributes_and_numbers(sentences, evidence, question, notes):
    """(b) outcome errors and (c) unsupported body/side/area/minute/yard claims."""
    outcome_errors, unsupported = [], []
    for sent in sentences:
        groups = citation_groups(sent)
        _scan_terms(sent, OUTCOME_TERMS, evidence, evidence_cats, "outcome", outcome_errors, question, label_cats=True)
        body_s = re.sub(BODY_TERMS["left foot"] + "|" + BODY_TERMS["right foot"], lambda m: " " * len(m.group(0)), sent, flags=re.I)
        _scan_terms(sent, BODY_TERMS, evidence, lambda e: {e["doc_attrs"]["body"]}, "body part", unsupported, question)
        _scan_terms(body_s, SIDE_TERMS, evidence, lambda e: {e["doc_attrs"]["side"]}, "side", unsupported, question)
        _scan_terms(sent, AREA_TERMS, evidence, lambda e: {e["doc_attrs"]["area"]}, "area", unsupported, question)
        # minutes and yards
        for pat in MINUTE_PATTERNS:
            for m in re.finditer(pat, sent, re.I):
                n = int(m.group(1))
                idx = items_for(sent, m.start(), m.end(), groups, len(evidence))
                pool = [evidence[i] for i in idx] if idx else evidence
                allowed = {e["minute"] for e in pool}
                if n not in allowed and str(n) not in re.findall(r"\d+", question):
                    unsupported.append({"type": "minute", "text": m.group(0), "sentence": sent})
        for m in YARDS.finditer(sent):
            n = int(m.group(1) or m.group(2))
            idx = items_for(sent, m.start(), m.end(), groups, len(evidence))
            pool = [evidence[i] for i in idx] if idx else evidence
            allowed = {e["doc_attrs"].get("yards") for e in pool}
            if m.group(2) or (n not in allowed and str(n) not in re.findall(r"\d+", question)):
                unsupported.append({"type": "distance", "text": m.group(0), "sentence": sent})
    return outcome_errors, unsupported


def check_names(answer, evidence, question):
    """Teams and player/other proper names that are not in the evidence (or the question)."""
    v = vocab()
    allowed = _words(question) | {w for e in evidence for w in _words(e["document_model"])}
    folded = fold_text(answer)
    out = []
    for m in v["team_re"].finditer(folded.casefold()):
        if not all(w in allowed for w in re.findall(r"\w+", m.group(0))):
            out.append({"type": "team", "text": folded[m.start():m.end()]})
    for sent in split_sentences(folded):
        body = CITE.sub(" ", sent)
        for m in re.finditer(r"[A-Z][\w'-]*", body):
            tok = re.match(r"[A-Za-z0-9À-ɏ]+", m.group(0))
            w = tok.group(0).casefold() if tok else ""
            first = body[:m.start()].strip(" .:;,\"'(") == ""
            if len(w) < 2 or w in allowed or w in ALLOWED_CAPS:
                continue
            if w in v["name_tokens"] or (not first and not m.group(0).isupper()):
                out.append({"type": "name", "text": m.group(0)})
    seen, uniq = set(), []
    for x in out:
        if (x["type"], x["text"].casefold()) not in seen:
            seen.add((x["type"], x["text"].casefold())); uniq.append(x)
    return uniq


def check_answer(example: dict, answer: str) -> dict:
    """Check `answer` against the example's evidence. example: a row of gen_inputs_*.jsonl (needs mode, query,
    evidence[document_model, doc_attrs, xg, outcome, minute], summary, retrieval.unsupported)."""
    answer = fold_text(answer).strip()
    evidence, question, mode = example["evidence"], fold_text(example["query"]), example["mode"]
    sentences = split_sentences(answer)
    reasons = []
    cites = [int(x) for x in CITE.findall(answer)]
    citations_ok = all(1 <= c <= len(evidence) for c in cites)
    if not citations_ok:
        reasons.append("citation index outside the evidence")
    if mode in ("answer", "partial") and not cites:
        citations_ok = False
        reasons.append("no [n] citation")
    if mode == "refuse" and cites:
        citations_ok = False
        reasons.append("citation in an answer that has no evidence")

    xg_ok, xg_details = check_xg(sentences, evidence, example["summary"].get("mean_xg"))
    if mode in ("answer", "partial") and not xg_details:
        xg_ok = False
        reasons.append("no xG value stated")
    if mode == "refuse" and xg_details:
        xg_ok = False
        reasons.append("xG stated although there is no evidence")
    if not xg_ok and xg_details:
        reasons.append("xG mismatch: " + ", ".join(str(d["stated"]) for d in xg_details if not d["ok"]))

    if evidence:
        outcome_errors, unsupported = check_attributes_and_numbers(sentences, evidence, question, example["retrieval"]["unsupported"])
    else:
        outcome_errors, unsupported = [], []
        for pat in MINUTE_PATTERNS:                                  # refusal: no minutes or yards beyond the question's own
            for m in re.finditer(pat, answer, re.I):
                if m.group(1) not in re.findall(r"\d+", question):
                    unsupported.append({"type": "minute", "text": m.group(0)})
    unsupported += check_names(answer, evidence, question)
    outcome_ok = not outcome_errors
    if outcome_errors:
        reasons.append("outcome: " + "; ".join(f"{e['text']!r}" for e in outcome_errors))
    if unsupported:
        reasons.append("unsupported: " + "; ".join(f"{u['type']}:{u['text']!r}" for u in unsupported))

    n_sent = len(sentences)
    format_ok = 2 <= n_sent <= 4 and len(answer.split()) <= 110
    if not format_ok:
        reasons.append(f"format: {n_sent} sentences, {len(answer.split())} words")
    low = answer.casefold()
    if mode == "refuse":
        missing = [w for w in ("team", "body", "side", "area", "outcome") if not re.search(rf"\b{w}", low)]
        if missing:
            format_ok = False
            reasons.append("refusal does not list supported attributes: missing " + ", ".join(missing))
    if mode == "partial":
        notes = example["retrieval"]["unsupported"]
        need = [w for w, pat in (("minute", r"minute|time|half"), ("player", r"player|name")) if any(w in n for n in notes) and not re.search(pat, low)]
        if need:
            format_ok = False
            reasons.append("partial answer does not mention the unsupported part: " + ", ".join(need))
    passed = xg_ok and outcome_ok and not unsupported and citations_ok and format_ok
    return {"passed": passed, "xg_ok": xg_ok, "xg_details": xg_details, "outcome_ok": outcome_ok, "outcome_errors": outcome_errors,
            "unsupported": unsupported, "citations_ok": citations_ok, "format_ok": format_ok, "reasons": reasons}
