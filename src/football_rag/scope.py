"""Query scope check: which of the 5 retriever attributes does the query mention, and is it in scope?

This is a KEYWORD LEXICON heuristic, separate from the bi-encoder (the model itself exposes no attributes).
It only says "the query contains words the retriever was trained on"; it does not verify the retriever
reads them correctly. Words come from the Phase 3 template phrases and phrase bank.
"""
import json
import re
from dataclasses import dataclass, field

from .config import RESULTS_P4

ATTRS = ("team", "body", "side", "area", "outcome")

BODY = re.compile(r"\b(?:left|right)[- ]foot(?:ed|ers?)?\b|\b(?:left|right) boot\b|\bhead(?:ed|er|ers)?\b")
SIDE = re.compile(r"\b(?:left|right)\b|\bcent(?:ral|re|er)\b|\bmiddle\b|\bflank\b|\bwing\b")
AREA = re.compile(r"\b(?:inside|outside|in|within) the (?:\d+-yard |penalty )?(?:box|area)\b|\bpenalty area\b|\b18-yard\b"
                  r"|\bclose range\b|\bfrom (?:distance|range)\b|\blong[- ]range\b|\blong (?:shots?|efforts?)\b|\bbox\b")
OUTCOME = re.compile(r"\bgoals?\b|\bscor(?:ed|es|ing)\b|\bsaved?\b|\bblock(?:ed)?\b|\bmiss(?:ed|es)?\b|\bwide\b|\boff target\b"
                     r"|\bover the bar\b|\bwent in\b|\bfound the net\b|\bbeat the keeper\b|\bkeeper\b|\bgoalkeeper\b|\bwayward\b"
                     r"|\bcharged down\b|\bpast the post\b|\bon target\b")
DOMAIN = re.compile(r"\b(?:shots?|attempts?|efforts?|chances?|strikes?)\b")   # a query "about shots" even with no attribute


@dataclass
class Parsed:
    attrs: dict = field(default_factory=dict)     # attribute -> matched text in the query
    has_domain: bool = False

    @property
    def n_attrs(self) -> int:
        return len(self.attrs)

    @property
    def known_words(self) -> bool:
        return bool(self.attrs) or self.has_domain

    @property
    def not_specified(self) -> list[str]:
        return [a for a in ATTRS if a not in self.attrs]


class Scope:
    def __init__(self, teams):
        names = set()
        for t in teams:
            names.add(t.casefold())
            if t.casefold().endswith(" women's"):
                names.add(t.casefold()[: -len(" women's")])      # "England" also names "England Women's"
        self.team_re = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)) + r")(?!\w)")

    def parse(self, query: str) -> Parsed:
        q, p = query.casefold(), Parsed()
        if m := self.team_re.search(q):
            p.attrs["team"] = m.group(0)
        if m := BODY.search(q):
            p.attrs["body"] = m.group(0)
            q_side = q[:m.start()] + " " + q[m.end():]            # "left foot" is body, not side
        else:
            q_side = q
        if m := SIDE.search(q_side):
            p.attrs["side"] = m.group(0)
        if m := AREA.search(q):
            p.attrs["area"] = m.group(0)
        if m := OUTCOME.search(q):
            p.attrs["outcome"] = m.group(0)
        p.has_domain = bool(DOMAIN.search(q))
        return p


def group_of(n_attrs: int) -> str:
    return "3+" if n_attrs >= 3 else str(n_attrs)


def decide(rule: dict, parsed: Parsed, top1: float, unsupported: list[str]) -> tuple[bool, bool, float]:
    """-> (low_confidence, withhold_evidence, threshold_used) for a scope rule.

    rule["name"]: "old" (single threshold), "per_attr_count" (a) or "lexicon_backstop" (b).
    Withholding: a minute/player cue in a query with no recognised attribute leaves nothing to search for
    (new rules). The old rule withheld when the cue came with low similarity.
    """
    name = rule["name"]
    if name == "old":
        thr = rule["backstop"]
        low = top1 < thr
        return low, bool(unsupported) and low, thr
    if name == "per_attr_count":
        thr = rule["group_thresholds"][group_of(parsed.n_attrs)]
        low = top1 < thr
    else:                                                          # lexicon_backstop
        thr = rule["backstop"]
        low = not parsed.known_words and top1 < thr
    return low, bool(unsupported) and parsed.n_attrs == 0, thr


def load_rule() -> dict:
    return json.loads((RESULTS_P4 / "scope_rule.json").read_text(encoding="utf-8"))
