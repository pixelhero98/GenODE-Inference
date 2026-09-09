"""Latent text-to-image clock comparison protocol."""

from genode.latent_clock.clocks import REFERENCE_CLOCK_KEYS, Clock, reference_clocks
from genode.latent_clock.contracts import BudgetLedger, ExecutionTrace, FrozenContext
from genode.latent_clock.protocol import BUDGETS, RewardScales, build_prompt_splits

__all__ = [
    "BUDGETS",
    "REFERENCE_CLOCK_KEYS",
    "BudgetLedger",
    "Clock",
    "ExecutionTrace",
    "FrozenContext",
    "RewardScales",
    "build_prompt_splits",
    "reference_clocks",
]
