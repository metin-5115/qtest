"""In-process telemetry sink for the ``--qtest-summary`` report.

Deliberately decoupled from the pytest plugin (no ``pytest`` import) so that
assertion code can record the distances it measures without pulling pytest into
the call path. The pytest plugin reads this log when rendering its end-of-session
summary; outside pytest the log simply accumulates and is otherwise inert.
"""

from __future__ import annotations


class _DistanceLog:
    """Accumulates ``(distance, metric, shots)`` samples recorded by assertions."""

    def __init__(self) -> None:
        self._samples: list[tuple[float, str | None, int | None]] = []

    def record(self, distance: float, metric: str | None = None, shots: int | None = None) -> None:
        self._samples.append((float(distance), metric, shots))

    def reset(self) -> None:
        self._samples.clear()

    @property
    def values(self) -> list[float]:
        return [d for d, _, _ in self._samples]

    @property
    def total_shots(self) -> int:
        """Sum of the shot counts actually used by the recorded assertions."""
        return sum(s for _, _, s in self._samples if s)

    def __len__(self) -> int:
        return len(self._samples)


_LOG = _DistanceLog()


def record_distance(distance: float, metric: str | None = None, shots: int | None = None) -> None:
    """Record a measured distance for the optional ``--qtest-summary`` report.

    Assertion helpers call this automatically; user tests may also call it to
    feed custom samples into the end-of-session report. *metric* labels the
    distance and *shots* records how many shots the measurement used so the
    summary can report the real shot total (rather than an estimate).
    """
    _LOG.record(distance, metric, shots)
