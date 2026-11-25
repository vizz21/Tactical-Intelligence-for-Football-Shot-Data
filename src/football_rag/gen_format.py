"""Phase 5: the fixed text format that the generator (flan-t5) reads, built from pipeline output.

The model never computes anything: xG, outcome, split and the top-k summary are precomputed. Shot ids (uuids)
and StatsBomb's own xG are NOT part of the input; the model cites [1], [2], [3] and the system attaches
the ids afterwards.
"""
import re
import unicodedata

K_EVIDENCE = 3
_SPECIAL = {"ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ł": "l", "Ł": "L", "æ": "ae", "Æ": "Ae", "ı": "i", "ð": "d", "Ð": "D",
            "þ": "th", "Þ": "Th"}


def fold_text(s: str) -> str:
    """ASCII-fold accents. flan-t5's vocabulary has no i-acute, n-tilde, c-acute, s-caron, ... (they become <unk>), and
    the model must copy names from its input, so the model sees (and writes) folded names, e.g. 'Rodriguez'.
    The system attaches the real shot ids / original names afterwards."""
    for a, b in _SPECIAL.items():
        s = s.replace(a, b)
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
SUPPORTED_ATTRIBUTES = "team, body part, side, area, outcome"
_OUTCOME = re.compile(r"Outcome: ([^.]+)\.$")


def outcome_word(document: str) -> str:
    """The outcome wording of the document text (e.g. 'saved', 'off target', 'hit the post', 'goal')."""
    return _OUTCOME.search(document).group(1)


def _cue_names(notes: list[str]) -> str:
    kinds = []
    if any("minute" in n for n in notes):
        kinds.append("minute")
    if any("player" in n for n in notes):
        kinds.append("player name")
    return " and ".join(kinds) or "unsupported detail"


def retrieval_status(summary: dict, withheld: bool) -> str:
    """Short, fixed status line derived from the pipeline's scope check."""
    notes = summary["unsupported"]
    if withheld:
        return f"not searched, unsupported: {_cue_names(notes)}"
    if summary["low_confidence"]:
        return "low confidence, query not recognised as a shot question"
    if notes:
        return f"ok, ignored unsupported part of the query: {_cue_names(notes)}"
    return "ok"


def evidence_for_model(result: dict) -> list[dict]:
    """Pipeline evidence -> the items shown to the model (none when the query is out of scope)."""
    if result["summary"]["low_confidence"] or not result["evidence"]:
        return []
    return result["evidence"][:K_EVIDENCE]


def format_input(query: str, items: list[dict], status: str) -> str:
    """One string: question, status, k numbered evidence items, precomputed top-k summary."""
    lines = [f"Question: {query}", f"Retrieval status: {status}"]
    if not items:
        lines.append("Evidence: none")
        return fold_text("\n".join(lines))
    lines.append("Evidence:")
    for i, e in enumerate(items, 1):
        lines.append(f"[{i}] {e['document']} | xG {e['xg']:.2f} | outcome: {outcome_word(e['document'])} | split: {e['split']}")
    mean_xg = sum(e["xg"] for e in items) / len(items)
    n_goals = sum(e["goal"] for e in items)
    lines.append(f"Top-{len(items)} summary: mean xG {mean_xg:.2f}; actual goals {n_goals} of {len(items)}.")
    return fold_text("\n".join(lines))
