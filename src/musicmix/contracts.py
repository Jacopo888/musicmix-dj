"""Contratti Python puri. I backend ML/DSP non sono dipendenze del dominio."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Mapping, Any


@dataclass(frozen=True)
class Feature:
    value: float | None
    confidence: float | None
    source: str = ''

    def __post_init__(self) -> None:
        for name, value in [('value', self.value), ('confidence', self.confidence)]:
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} deve essere tra 0 e 1')


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    track_a: str
    track_b: str
    cue_out_sample: int
    cue_in_sample: int
    mode: str
    features: Mapping[str, Feature]
    penalties: Mapping[str, Feature]
    passes_constraints: bool = True
    constraint_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.track_a or not self.track_b:
            raise ValueError('Identificatori obbligatori')
        if self.track_a == self.track_b:
            raise ValueError('Questo prototipo richiede due tracce differenti')
        if self.cue_out_sample < 0 or self.cue_in_sample < 0:
            raise ValueError('Cue samples non negativi')


@dataclass(frozen=True)
class ScoreReport:
    candidate_id: str
    valid: bool
    score: float | None
    contributions: Mapping[str, float]
    penalties: Mapping[str, float]
    missing: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


class FeatureProvider(Protocol):
    def compute(self, candidate: Candidate) -> Mapping[str, Feature]: ...


class CandidateGenerator(Protocol):
    def generate(self, track_a: str, track_b: str, context: Any) -> list[Candidate]: ...


class Ranker(Protocol):
    def score(self, candidate: Candidate) -> ScoreReport: ...


class Renderer(Protocol):
    def render(self, candidate: Candidate, options: Mapping[str, Any]) -> str: ...
