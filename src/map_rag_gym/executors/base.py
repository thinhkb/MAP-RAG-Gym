from __future__ import annotations

from typing import Any, Dict

from map_rag_gym.core.schemas import PipelineState, StepRecord


class Executor:
    name = "BASE"

    def run(self, state: PipelineState, **kwargs: Any) -> StepRecord:
        raise NotImplementedError

    def step(self, state: PipelineState, input_data: Dict[str, Any], output_data: Dict[str, Any], cost: Dict[str, float] | None = None, notes: str = "") -> StepRecord:
        return StepRecord(
            step_id=state.next_step_id(),
            module=self.name,
            input_data=input_data,
            output_data=output_data,
            cost=cost or {},
            notes=notes,
        )
