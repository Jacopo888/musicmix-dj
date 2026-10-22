"""Ranker interpretativo di PROTOTIPO, richiede feature fornite esternamente.

Nessun valore è una misura fisica provata: pesi e soglie sono ipotesi iniziali.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .contracts import Candidate, ScoreReport


@dataclass(frozen=True)
class ScoreConfig:
    weights: dict[str, float] = field(default_factory=lambda: {
        'horizontal': 0.40, 'vertical': 0.10, 'continuity': 0.35, 'creative': 0.15})
    penalty_weights: dict[str, float] = field(default_factory=lambda: {
        'transform': 0.12, 'artifact': 0.12})
    min_confidence: float = 0.50

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError('min_confidence fuori range')
        if any(w < 0 for w in (*self.weights.values(), *self.penalty_weights.values())):
            raise ValueError('Non sono ammessi pesi negativi')


class InterpretableRanker:
    def __init__(self, config: ScoreConfig | None = None):
        self.config = config or ScoreConfig()

    def score(self, candidate: Candidate) -> ScoreReport:
        if not candidate.passes_constraints:
            return ScoreReport(candidate.candidate_id, False, None, {}, {}, (),
                               ('hard_constraint_failed', *candidate.constraint_reasons))

        numerator = 0.0
        denominator = 0.0
        raw = {}
        missing = []
        for name, weight in self.config.weights.items():
            f = candidate.features.get(name)
            if f is None or f.value is None or f.confidence is None or f.confidence < self.config.min_confidence:
                missing.append(name)
                continue
            numerator += weight * f.value
            denominator += weight
            raw[name] = (f.value, weight)

        if denominator <= 0:
            return ScoreReport(candidate.candidate_id, False, None, {}, {},
                               tuple(missing), ('no_reliable_benefit_features',))

        contributions = {n: round(v*w/denominator, 6) for n,(v,w) in raw.items()}
        penalties = {}
        for name, weight in self.config.penalty_weights.items():
            f = candidate.penalties.get(name)
            if f is None or f.value is None or f.confidence is None or f.confidence < self.config.min_confidence:
                missing.append(f'penalty:{name}')
                continue
            penalties[name] = round(weight*f.value, 6)

        score = min(1.0, max(0.0, numerator / denominator - sum(penalties.values())))
        notes = ('Score euristico di prototipo: non equivale a qualità musicale validata.',)
        return ScoreReport(candidate.candidate_id, True, round(score, 6),
                           contributions, penalties, tuple(missing), notes)
