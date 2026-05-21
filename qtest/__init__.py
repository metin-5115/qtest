"""qtest — Statistical, pytest-native testing for quantum circuits.

qtest brings the discipline of modern software testing to quantum programs.
It plugs into pytest, provides statistical assertions designed for the noisy,
probabilistic outputs of quantum measurements, and integrates with Hypothesis
to enable property-based testing for quantum circuits.

Public API (to be re-exported as the package grows):
    - assert_distribution_close
    - assert_state_close
    - assert_unitary
    - assert_circuit_equivalent

See https://qtest.readthedocs.io for full documentation.
"""

__version__ = "0.1.0"
