from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import TextIO

import torch


def default_torch_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_torch_device(requested: str | None = None) -> torch.device:
    value = str(requested or "auto").strip().lower()
    if value in {"", "auto"}:
        return torch.device(default_torch_device())
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {requested!r}, but CUDA is not available.")
    return torch.device(value)


@dataclass
class ProgressBar:
    total: int
    label: str
    enabled: bool = True
    stream: TextIO = sys.stderr
    min_interval_s: float = 1.0

    def __post_init__(self) -> None:
        self.total = max(0, int(self.total))
        self.count = 0
        self._last_emit = 0.0
        self._started = False
        self._tty = bool(getattr(self.stream, "isatty", lambda: False)())

    def __enter__(self) -> "ProgressBar":
        if self.enabled and self.total > 0:
            self._emit(force=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.enabled and self.total > 0:
            self.count = self.total if exc_type is None else self.count
            self._emit(force=True)
            self.stream.write("\n")
            self.stream.flush()

    def update(self, step: int = 1) -> None:
        if not self.enabled or self.total <= 0:
            return
        self.count = min(self.total, self.count + int(step))
        self._emit(force=self.count >= self.total)

    def _emit(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_emit < float(self.min_interval_s):
            return
        self._last_emit = now
        self._started = True
        ratio = float(self.count) / float(max(1, self.total))
        if self._tty:
            width = 28
            filled = int(round(width * ratio))
            bar = "#" * filled + "-" * (width - filled)
            self.stream.write(f"\r{self.label}: [{bar}] {self.count}/{self.total}")
        else:
            self.stream.write(f"{self.label}: {self.count}/{self.total}\n")
        self.stream.flush()
