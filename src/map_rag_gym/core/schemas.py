from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List
import time
import uuid


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepRecord:
    step_id: int
    module: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    scores: Dict[str, float | None] = field(default_factory=dict)
    cost: Dict[str, float] = field(default_factory=dict)
    notes: str = ""


@dataclass
class RunResult:
    run_id: str
    question: str
    gold_answer: str
    workflow_id: str
    planner_reason: str
    final_answer: str
    final_scores: Dict[str, float]
    total_cost: Dict[str, float]
    steps: List[StepRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = time.time()
        return payload


@dataclass
class PlannerDecision:
    workflow_id: str
    confidence: float
    reason: str


@dataclass
class QuestionFeatures:
    question: str
    token_len: int
    comparative_flag: int
    conjunction_flag: int
    ambiguity_flag: int
    temporal_flag: int = 0
    negation_flag: int = 0
    superlative_flag: int = 0
    multi_entity_flag: int = 0
    entity_density: float = 0.0
    wh_word: str = "unknown"
    estimated_hops: int = 1


@dataclass
class PipelineState:
    question: str
    gold_answer: str = ""
    workflow_id: str = ""
    history: List[StepRecord] = field(default_factory=list)
    working_memory: Dict[str, Any] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = field(default_factory=time.time)

    def add_step(self, step: StepRecord) -> None:
        self.history.append(step)

    def next_step_id(self) -> int:
        return len(self.history) + 1
