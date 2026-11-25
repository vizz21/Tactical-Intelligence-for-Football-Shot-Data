"""Core Phase 4 pipeline: query -> bi-encoder retrieval -> xG + outcome -> evidence with source pointers.

The retriever only understands 5 attributes (team, body part, side, area, outcome). Minute, a named player
and build-up questions were never trained; `detect_unsupported` is a deliberately minimal heuristic that
spots minute patterns and player names so the caller gets a clear message instead of silent nonsense.
"""
import json
import re
from functools import lru_cache

import numpy as np
import pandas as pd

from . import retrieval as R
from .config import OPEN_DATA_EVENTS, SOURCE_TEMPLATE, TABULAR, XG_SCORES
from .scope import Scope, decide, load_rule

SUPPORTED_NOTE = "Supported query attributes: team, body part, side (left/right/central), area (inside/outside the box), outcome (goal/saved/blocked/missed)."
MINUTE_PATTERNS = [r"\bminutes?\b", r"\bmins?\b", r"\b\d{1,3}\s?(st|nd|rd|th)\b", r"\b\d{1,3}'", r"\b(first|second)\s+half\b"]
# words that also occur in player names but are ordinary query vocabulary (kept out of the player-name check)
COMMON_WORDS = {"shot", "shots", "goal", "goals", "right", "left", "head", "foot", "box", "side", "outside", "inside",
                "central", "saved", "blocked", "missed", "scored", "header", "headers", "centre", "center", "range",
                "distance", "keeper", "goalkeeper", "target", "attempts", "efforts", "find", "show", "search", "wide",
                "long", "close", "women", "player", "from", "with", "that", "were", "have", "kept", "held", "stopped"}
TOKEN = re.compile(r"\w+", re.UNICODE)


def _norm(s: str) -> str:
    return s.casefold()


def _clean(v):
    """numpy/pandas scalar -> plain python value (NaN -> None) for JSON."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


class Pipeline:
    """Loads encoder, index, xG scores and shot metadata once; search() and get_shot() are then cheap."""

    def __init__(self, device: str = "cpu", rule: dict | None = None):
        """rule: scope rule (see scope.decide); default is the one chosen in results/phase4/scope_rule.json."""
        self.corpus = R.load_corpus()
        self.index = R.load_index()
        self.encoder = R.load_encoder(device)
        tab = pd.read_csv(TABULAR)
        scores = pd.read_csv(XG_SCORES)
        assert (tab["row_idx"].to_numpy() == np.arange(len(tab))).all()
        assert (scores["row_idx"].to_numpy() == np.arange(len(scores))).all()
        assert (scores["shot_id"] == tab["shot_id"]).all() and self.index.shot_ids == list(tab["shot_id"]), "index/xG/tabular order mismatch"
        self.tab = tab
        self.xg = scores["xg"].to_numpy()
        self.in_sample = scores["in_sample"].to_numpy().astype(bool)
        self.goal = tab["goal"].to_numpy().astype(int)
        self.shot_pos = {s: i for i, s in enumerate(tab["shot_id"])}
        self.rule = rule or load_rule()
        self.threshold = self.rule["backstop"]
        self.split_arr = tab["split"].to_numpy()
        self.is_test = self.split_arr == "test"
        self.scope = Scope(tab["team"].unique())
        self._build_name_vocab()

    # ---------------------------------------------------------------- unsupported-question heuristic
    def _build_name_vocab(self):
        team_tokens = {_norm(t) for team in self.tab["team"].unique() for t in TOKEN.findall(team)}
        self.full_names = {_norm(p) for p in self.tab["player"].dropna().unique()}
        toks = {_norm(t) for p in self.tab["player"].dropna().unique() for t in TOKEN.findall(p)}
        self.name_tokens = {t for t in toks if len(t) >= 5 and t not in team_tokens and t not in COMMON_WORDS}

    def detect_unsupported(self, query: str) -> list[str]:
        """Heuristic only: returns human-readable notes for minute patterns and player names found in the query."""
        notes, q = [], _norm(query)
        if any(re.search(p, q) for p in MINUTE_PATTERNS):
            notes.append("minute/time of the match is not supported by the retriever (it only learned 5 attributes)")
        if any(n in q for n in self.full_names) or any(t in set(TOKEN.findall(q)) for t in self.name_tokens):
            notes.append("player names are not supported by the retriever (it only learned 5 attributes)")
        return notes

    # ---------------------------------------------------------------- evidence
    def _source(self, i: int) -> dict:
        match_id = int(self.tab.at[i, "match_id"])
        return {"file": SOURCE_TEMPLATE.format(match_id=match_id), "event_id": self.tab.at[i, "shot_id"]}

    def _item(self, i: int, rank: int | None = None, score: float | None = None) -> dict:
        row, doc = self.tab.iloc[i], self.corpus[i]
        item = {"rank": rank, "score": score, "shot_id": row["shot_id"], "match_id": int(row["match_id"]),
                "team": row["team"], "player": _clean(row["player"]), "minute": int(row["minute"]), "period": int(row["period"]),
                "document": doc["document"], "doc_attrs": doc["doc_attrs"], "split": row["split"],
                "in_sample": bool(self.in_sample[i]), "xg": float(self.xg[i]), "goal": int(self.goal[i]),
                "source": self._source(i)}
        return item

    def search(self, query: str, k: int = 10, test_only: bool = False, splits=None) -> dict:
        """Top-k shots for a query. test_only restricts the candidates to test-split shots (the top-k is filled
        from the filtered pool, so it is still k long). `splits` (internal, e.g. {"train"}, used to build the
        Phase 5 data) generalises it to any set of splits and overrides test_only. The scope check always uses
        the best similarity over the WHOLE index, so it does not depend on the filter."""
        k = int(max(1, min(k, 100)))
        notes = self.detect_unsupported(query)
        parsed = self.scope.parse(query)
        sims = self.index.similarities(R.embed(self.encoder, [query]))[0]
        top1_all = float(sims.max())
        low_conf, withhold, thr = decide(self.rule, parsed, top1_all, notes)

        if splits is not None:
            splits = set(splits)
            assert splits <= {"train", "val", "test"}, f"unknown split in {splits}"
            pool = np.isin(self.split_arr, list(splits))
        else:
            pool = self.is_test if test_only else np.ones(len(sims), bool)
        masked = np.where(pool, sims, -np.inf)
        kk = min(k, int(pool.sum()))
        top = np.argpartition(-masked, kk - 1)[:kk]
        top = top[np.argsort(-masked[top])]
        above = (sims >= thr) & pool

        messages = [f"Not supported: {n}." for n in notes]
        if low_conf:
            messages.append(f"Low confidence: the query contains no words the retriever knows and its best similarity {top1_all:.2f} is "
                            f"below {thr:.2f}; it may be outside what the retriever understands. " + SUPPORTED_NOTE)
        elif notes:
            messages.append("The unsupported part of the query is ignored; results reflect only the supported attributes. " + SUPPORTED_NOTE)
        elif not parsed.attrs:
            messages.append("No specific attribute recognised (team, body part, side, area, outcome): results are the most similar shots overall.")

        evidence = [] if withhold else [self._item(int(i), r + 1, float(sims[i])) for r, i in enumerate(top)]
        n_above = int(above.sum())
        summary = {
            "n_results": len(evidence),
            "mean_xg": float(np.mean([e["xg"] for e in evidence])) if evidence else None,
            "n_goals": int(sum(e["goal"] for e in evidence)),
            "top_score": float(sims[top[0]]),
            "threshold": float(thr),
            "low_confidence": low_conf,
            "n_above_threshold": n_above,
            "mean_xg_above_threshold": float(self.xg[above].mean()) if n_above else None,
            "n_goals_above_threshold": int(self.goal[above].sum()),
            "understood": parsed.attrs,
            "not_specified": parsed.not_specified,
            "test_only": bool(splits == {"test"}) if splits is not None else test_only,
            "searched_splits": sorted(splits) if splits is not None else (["test"] if test_only else ["train", "val", "test"]),
            "unsupported": notes,
            "messages": messages,
        }
        return {"query": query, "k": k, "evidence": evidence, "summary": summary}

    # ---------------------------------------------------------------- single shot
    def get_shot(self, shot_id: str) -> dict | None:
        i = self.shot_pos.get(shot_id)
        if i is None:
            return None
        item = self._item(i)
        feats = self.tab.iloc[i].drop(["row_idx", "shot_id"]).to_dict()
        item["features"] = {k: _clean(v) for k, v in feats.items()}
        item["source_event"] = self.load_source_event(int(self.tab.at[i, "match_id"]), shot_id)
        item.pop("rank"), item.pop("score")
        return item

    @staticmethod
    @lru_cache(maxsize=8)
    def _events(match_id: int):
        path = OPEN_DATA_EVENTS / f"{match_id}.json"
        if not path.exists():
            return None
        return {e["id"]: e for e in json.loads(path.read_text(encoding="utf-8"))}

    def load_source_event(self, match_id: int, event_id: str) -> dict | None:
        """The unmodified StatsBomb event (None if the raw open-data folder is not present on this machine)."""
        ev = self._events(match_id)
        return None if ev is None else ev.get(event_id)
