"""High-precision latency instrumentation and statistics."""

from __future__ import annotations

import time
import numpy as np
from typing import Dict, List
from pydantic import BaseModel, Field


class LatencyStats(BaseModel):
    """Statistical summary of execution latencies in milliseconds."""
    count: int
    mean_ms: float
    std_ms: float
    min_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float

    @classmethod
    def calculate(cls, latencies_ms: List[float]) -> LatencyStats:
        """Calculates percentile distribution from recorded millisecond latencies."""
        if not latencies_ms:
            return cls(
                count=0,
                mean_ms=0.0,
                std_ms=0.0,
                min_ms=0.0,
                p50_ms=0.0,
                p95_ms=0.0,
                p99_ms=0.0,
                max_ms=0.0,
            )

        arr = np.array(latencies_ms)
        return cls(
            count=len(latencies_ms),
            mean_ms=float(np.mean(arr)),
            std_ms=float(np.std(arr)),
            min_ms=float(np.min(arr)),
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            max_ms=float(np.max(arr)),
        )


class LatencyTracker:
    """Tracks latency across multiple phases of retrieval execution."""

    def __init__(self):
        self.measurements: Dict[str, List[float]] = {}

    def record(self, phase: str, duration_ms: float) -> None:
        self.measurements.setdefault(phase, []).append(duration_ms)

    def get_stats(self, phase: str) -> LatencyStats:
        return LatencyStats.calculate(self.measurements.get(phase, []))

    def get_all_stats(self) -> Dict[str, LatencyStats]:
        return {phase: LatencyStats.calculate(vals) for phase, vals in self.measurements.items()}
