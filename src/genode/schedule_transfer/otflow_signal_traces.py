from __future__ import annotations
from typing import Sequence, Tuple
import numpy as np
import torch
NATIVE_INFO_GROWTH_ROW_KEY = "info_growth_hardness"
NATIVE_INFO_GROWTH_TRACE_KEY = "info_growth_hardness_by_step"
BASE_MODEL_SIGNAL_SPECS: Tuple[Tuple[str, str], ...] = (
    ("disagreement", "disagreement_by_step"), ("residual_norm", "residual_norm_by_step"), ("hybrid_signal", "hybrid_signal_by_step"),
    ("u_disagreement", "u_disagreement_by_step"), ("u_residual_norm", "u_residual_norm_by_step"), ("u_hybrid_signal", "u_hybrid_signal_by_step"),
    ("variance_scaled_signal", "variance_scaled_signal_by_step"), ("top_book_disagreement", "top_book_disagreement_by_step"),
    ("top_book_residual_norm", "top_book_residual_norm_by_step"), ("top_book_hybrid_signal", "top_book_hybrid_signal_by_step"),
)
MODEL_SIGNAL_SPECS: Tuple[Tuple[str, str], ...] = BASE_MODEL_SIGNAL_SPECS
NATIVE_SIGNAL_TRACE_KEYS: Tuple[str, ...] = tuple(out_key for _, out_key in MODEL_SIGNAL_SPECS) + (NATIVE_INFO_GROWTH_TRACE_KEY,)
def resolved_info_growth_scale(residual_norm_values: Sequence[float]) -> float:
    residual=np.asarray(residual_norm_values,dtype=np.float64); residual=residual[np.isfinite(residual)]
    return 1.0 if residual.size==0 else max(float(np.mean(np.clip(residual,0.0,None))),1e-8)
def compute_info_growth_hardness(residual_norm, disagreement, *, scale: float):
    if float(scale)<=0.0: raise ValueError(f"scale must be positive, got {scale}")
    return disagreement * torch.log1p(residual_norm / float(scale))
def compute_info_growth_hardness_numpy(residual_norm: Sequence[float] | np.ndarray, disagreement: Sequence[float] | np.ndarray, *, scale: float) -> np.ndarray:
    if float(scale)<=0.0: raise ValueError(f"scale must be positive, got {scale}")
    return np.asarray(disagreement,dtype=np.float64) * np.log1p(np.clip(np.asarray(residual_norm,dtype=np.float64),0.0,None)/float(scale))
__all__=["BASE_MODEL_SIGNAL_SPECS","MODEL_SIGNAL_SPECS","NATIVE_INFO_GROWTH_ROW_KEY","NATIVE_INFO_GROWTH_TRACE_KEY","NATIVE_SIGNAL_TRACE_KEYS","compute_info_growth_hardness","compute_info_growth_hardness_numpy","resolved_info_growth_scale"]
