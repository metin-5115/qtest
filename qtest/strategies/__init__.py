"""Hypothesis strategies for property-based testing of quantum code.

The strategies in this subpackage are designed to be drop-in inputs to
:func:`hypothesis.given`. They cover the three things property tests
about quantum software most often need:

* whole **circuits** (:func:`quantum_circuits`),
* individual **gate descriptors** or **Pauli operators**
  (:func:`random_gates`, :func:`pauli_strings`),
* **state vectors** and **density matrices**
  (:func:`random_states`, :func:`product_states`,
  :func:`random_density_matrices`).

Hypothesis (and its companion ``hypothesis[numpy]`` extra) is an
**optional** qtest dependency. Importing this subpackage when
Hypothesis is not installed raises :class:`ImportError` at import time
so users get a clear, early error instead of a confusing traceback the
first time they call ``@given``.
"""

from __future__ import annotations

from importlib.util import find_spec

if find_spec("hypothesis") is None:
    raise ImportError(
        "qtest.strategies requires the 'hypothesis' package. "
        "Install with: pip install 'qtest[hypothesis]'"
    )

from qtest.strategies.circuits import quantum_circuits
from qtest.strategies.gates import (
    DEFAULT_GATE_SET,
    pauli_strings,
    random_gates,
)
from qtest.strategies.states import (
    product_states,
    random_density_matrices,
    random_states,
)

__all__ = [
    "DEFAULT_GATE_SET",
    "pauli_strings",
    "product_states",
    "quantum_circuits",
    "random_density_matrices",
    "random_gates",
    "random_states",
]
