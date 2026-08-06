"""Per-call cost tracking (M6). Kept as its own module so pricing can change
without touching the batch job or vision code."""
from __future__ import annotations

from sqlmodel import Session

from .models import CostLedger

# USD per token, illustrative rates - swap for real published pricing as needed.
PRICING = {
    "vision": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
    "embedding": {"input": 0.02 / 1_000_000, "output": 0.0},
}


def estimate_cost(call_type: str, input_tokens: int, output_tokens: int) -> float:
    rates = PRICING.get(call_type, {"input": 0.0, "output": 0.0})
    return input_tokens * rates["input"] + output_tokens * rates["output"]


def record_cost(session: Session, call_type: str, ref_table: str, ref_id: int,
                 input_tokens: int, output_tokens: int) -> CostLedger:
    entry = CostLedger(
        call_type=call_type, ref_table=ref_table, ref_id=ref_id,
        input_tokens=input_tokens, output_tokens=output_tokens,
        estimated_cost_usd=estimate_cost(call_type, input_tokens, output_tokens),
    )
    session.add(entry)
    session.commit()
    session.refresh(entry)
    return entry


def total_cost(session: Session) -> float:
    from sqlmodel import select
    return sum(c.estimated_cost_usd for c in session.exec(select(CostLedger)).all())
