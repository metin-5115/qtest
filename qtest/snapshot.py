"""Snapshot (golden-file) testing for measurement distributions.

The first time a snapshot test runs (or whenever ``--qtest-snapshot-update`` is
passed) qtest samples the circuit and writes the resulting distribution to a
JSON *golden file* next to your test. On later runs it re-samples and compares
against the stored baseline with the usual statistical metric — so you can lock
in a circuit's behaviour and get alerted when a refactor changes it, without
hand-writing the expected distribution.

Use it through the :func:`qtest_snapshot` pytest fixture::

    def test_my_circuit(qtest_snapshot):
        qc = build_circuit()
        qc.measure_all()
        qtest_snapshot.assert_distribution_close(qc, shots=4096, tolerance=0.1)

Golden files live in a ``__qtest_snapshots__/`` directory beside the test file
and should be committed to version control. Refresh them deliberately with::

    pytest --qtest-snapshot-update
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_SNAPSHOT_VERSION = 1
_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _sanitize(name: str) -> str:
    """Make *name* safe to use as a filename component."""
    cleaned = _SANITIZE_RE.sub("_", name).strip("_")
    return cleaned or "snapshot"


class Snapshot:
    """A handle to the golden files for one test.

    Usually obtained from the :func:`qtest_snapshot` fixture rather than
    constructed directly.

    Parameters
    ----------
    directory
        Directory holding this test's golden files (created on demand).
    default_name
        Snapshot name used when a call does not pass an explicit ``name``
        (typically derived from the test's node id).
    update
        When ``True`` (``--qtest-snapshot-update``), golden files are
        (re)written instead of compared.
    """

    def __init__(self, directory: Path, default_name: str, update: bool = False) -> None:
        self.directory = Path(directory)
        self.default_name = default_name
        self.update = update

    def path_for(self, name: str | None = None) -> Path:
        """Return the golden-file path for *name* (or the default name)."""
        return self.directory / f"{_sanitize(name or self.default_name)}.json"

    def assert_distribution_close(
        self,
        circuit: Any,
        name: str | None = None,
        shots: int | None = None,
        tolerance: float | None = None,
        metric: str | None = None,
        backend: Any | None = None,
        seed: int | None = None,
        noise_model: Any | None = None,
        msg: str | None = None,
    ) -> None:
        """Compare *circuit*'s distribution against a stored snapshot.

        On the first run (no golden file yet) or under ``--qtest-snapshot-update``
        the measured distribution is written to disk and the check passes.
        Otherwise the circuit is re-sampled and compared to the stored
        distribution via :func:`qtest.assert_distribution_close`.

        Parameters
        ----------
        circuit
            A backend-native circuit with measurements.
        name
            Snapshot name (allows multiple snapshots per test). Defaults to the
            test's node-id-derived name.
        shots, tolerance, metric, backend, seed, noise_model, msg
            Forwarded to sampling / comparison; ``None`` means "use config".
        """
        from qtest.assertions.distribution import assert_distribution_close
        from qtest.config import _resolve_value

        path = self.path_for(name)

        if self.update or not path.exists():
            measured = self._sample(circuit, shots, backend, seed, noise_model)
            self._write(path, measured, metric=_resolve_value("statistical_metric", metric))
            return

        stored = self._read(path)
        assert_distribution_close(
            circuit,
            expected=stored["distribution"],
            shots=shots,
            tolerance=tolerance,
            metric=metric,
            backend=backend,
            seed=seed,
            noise_model=noise_model,
            msg=msg or f"Snapshot mismatch vs {path.name}",
        )

    # ------------------------------------------------------------------ #
    # Internals                                                           #
    # ------------------------------------------------------------------ #

    def _sample(
        self,
        circuit: Any,
        shots: int | None,
        backend: Any | None,
        seed: int | None,
        noise_model: Any | None,
    ) -> dict[str, float]:
        """Sample *circuit* and return its normalised distribution."""
        from qtest.assertions.distribution import _has_measurements, _normalize_counts
        from qtest.backends.registry import get_backend
        from qtest.config import _resolve_value, get_config

        shots = _resolve_value("default_shots", shots)
        seed = _resolve_value("default_seed", seed)
        if not _has_measurements(circuit):
            raise ValueError(
                "circuit has no measurement instructions; snapshot tests require "
                "a circuit that produces classical bitstrings."
            )
        if backend is None:
            backend = get_backend(get_config().default_backend)
        run_kwargs: dict[str, Any] = {"shots": shots, "seed": seed}
        if noise_model is not None:
            run_kwargs["noise_model"] = noise_model
        raw = backend.run_circuit(circuit, **run_kwargs)
        counts = {k.replace(" ", ""): v for k, v in raw.items()}
        return _normalize_counts(counts)

    def _write(self, path: Path, distribution: dict[str, float], metric: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "qtest_snapshot_version": _SNAPSHOT_VERSION,
            "metric": metric,
            "distribution": {k: distribution[k] for k in sorted(distribution)},
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _read(self, path: Path) -> dict[str, Any]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "distribution" not in data:
            raise ValueError(f"Malformed qtest snapshot file: {path}")
        return data
