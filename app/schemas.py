"""Pydantic request/response models for the API."""
from typing import Any

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, examples=["headers from the left that were saved"])
    k: int = Field(10, ge=1, le=100)
    test_only: bool = Field(False, description="search only test-split shots (xG not trained on them); top-k is filled from the filtered pool")


class Source(BaseModel):
    file: str = Field(description="StatsBomb events file, relative to the project root")
    event_id: str = Field(description="id of the shot event inside that file (equals shot_id)")


class Evidence(BaseModel):
    rank: int
    score: float = Field(description="cosine similarity between query and document")
    shot_id: str
    match_id: int
    team: str
    player: str | None
    minute: int
    period: int
    document: str
    doc_attrs: dict[str, Any]
    split: str
    in_sample: bool = Field(description="True if the xG model was trained on this shot (train split)")
    xg: float
    goal: int = Field(description="actual outcome: 1 = goal")
    source: Source


class Summary(BaseModel):
    n_results: int
    mean_xg: float | None
    n_goals: int
    top_score: float
    threshold: float
    low_confidence: bool
    n_above_threshold: int = Field(description="documents in the searched pool with similarity >= threshold (a similarity cut-off, not a count of matching shots)")
    mean_xg_above_threshold: float | None
    n_goals_above_threshold: int
    understood: dict[str, str] = Field(description="attributes found in the query by a keyword lexicon (heuristic, not the model's own reading): attribute -> matched text")
    not_specified: list[str] = Field(description="attributes (team, body, side, area, outcome) not found in the query")
    test_only: bool
    unsupported: list[str] = Field(description="heuristic notes: minute pattern or player name found in the query")
    messages: list[str]


class SearchResponse(BaseModel):
    query: str
    k: int
    evidence: list[Evidence]
    summary: Summary


class ShotDetail(BaseModel):
    shot_id: str
    match_id: int
    team: str
    player: str | None
    minute: int
    period: int
    document: str
    doc_attrs: dict[str, Any]
    split: str
    in_sample: bool
    xg: float
    goal: int
    source: Source
    features: dict[str, Any] = Field(description="the tabular row (ANN inputs plus metadata)")
    source_event: dict[str, Any] | None = Field(description="unmodified StatsBomb event, null if open-data/ is not on this machine")


class AnswerRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, examples=["Spain Women's left-footed shots from inside the box"])
    test_only: bool = Field(True, description="evidence from test-split shots only (the setting the generator was evaluated in); "
                                              "with False the three shots may come from the training split, where the xG is in-sample")


class AnswerResponse(BaseModel):
    query: str
    answer: str = Field(description="text shown to the user: the model's answer if it passed all checks, otherwise a rule-based template")
    source: str = Field(description="'model' = fine-tuned flan-t5 text that passed the checker and group rules; 'template' = deterministic fallback")
    fallback_reasons: list[str] = Field(description="why the model text was rejected (empty when source is 'model')")
    raw_model_answer: str = Field(description="what the model wrote before verification (for transparency; may be wrong)")
    citations: dict[str, str] = Field(description="[n] -> shot_id of the evidence item the answer cites")
    evidence: list[Evidence]
    summary: Summary
    notice: str


class Health(BaseModel):
    status: str
    n_documents: int
    n_shots: int
    xg_model: str
    encoder: str
    low_confidence_threshold: float = Field(description="similarity backstop of the scope rule")
    scope_rule: str
