"""Project-wide pytest configuration for the qtest test suite."""

# Make qtest's built-in fixtures available throughout the test tree.
pytest_plugins = [
    "qtest.fixtures.common_states",
    "qtest.fixtures.common_gates",
    "qtest.fixtures.noise",
]
