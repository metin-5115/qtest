"""Tests for the backend registry."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from qtest.backends import registry
from qtest.backends.base import Backend

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


class _DummyBackend(Backend):
    """Minimal concrete Backend for registry tests."""

    def run_circuit(
        self, circuit: Any, shots: int | None = None, seed: int | None = None
    ) -> dict[str, int]:
        return {"0": shots or 1}

    def get_statevector(self, circuit: Any) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=complex)

    def get_unitary(self, circuit: Any) -> np.ndarray:
        return np.eye(2, dtype=complex)

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def supports_statevector(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def reset_registry_state():
    """Snapshot registry state before each test and restore after."""
    saved = registry._REGISTRY.copy()
    saved_default = registry._DEFAULT_BACKEND_NAME
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(saved)
    registry._DEFAULT_BACKEND_NAME = saved_default


# --------------------------------------------------------------------------- #
# Auto-registration                                                           #
# --------------------------------------------------------------------------- #


def test_qiskit_registered_by_default() -> None:
    assert "qiskit" in registry.list_available_backends()


def test_default_backend_is_qiskit_initially() -> None:
    assert registry._DEFAULT_BACKEND_NAME == "qiskit"


# --------------------------------------------------------------------------- #
# register_backend                                                            #
# --------------------------------------------------------------------------- #


def test_register_and_get_returns_instance() -> None:
    registry.register_backend("dummy", _DummyBackend)
    b = registry.get_backend("dummy")
    assert isinstance(b, _DummyBackend)


def test_register_returns_fresh_instances() -> None:
    """Each get_backend call should produce a new instance (no shared state)."""
    registry.register_backend("dummy", _DummyBackend)
    a = registry.get_backend("dummy")
    b = registry.get_backend("dummy")
    assert a is not b


def test_register_overwrites_previous_entry() -> None:
    registry.register_backend("dummy", _DummyBackend)

    class _OtherDummy(_DummyBackend):
        @property
        def name(self) -> str:
            return "other-dummy"

    registry.register_backend("dummy", _OtherDummy)
    b = registry.get_backend("dummy")
    assert b.name == "other-dummy"


def test_register_rejects_non_backend_class() -> None:
    class NotABackend:
        pass

    with pytest.raises(TypeError):
        registry.register_backend("bad", NotABackend)  # type: ignore[arg-type]


def test_register_rejects_non_class() -> None:
    with pytest.raises(TypeError):
        registry.register_backend("bad", _DummyBackend())  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_name", ["", None, 42])
def test_register_rejects_bad_name(bad_name: Any) -> None:
    with pytest.raises(ValueError):
        registry.register_backend(bad_name, _DummyBackend)


# --------------------------------------------------------------------------- #
# get_backend                                                                 #
# --------------------------------------------------------------------------- #


def test_get_backend_unknown_name_raises_key_error() -> None:
    with pytest.raises(KeyError):
        registry.get_backend("does-not-exist")


def test_get_backend_unknown_lists_available() -> None:
    registry.register_backend("dummy", _DummyBackend)
    with pytest.raises(KeyError, match="dummy"):
        registry.get_backend("does-not-exist")


def test_get_backend_without_default_raises() -> None:
    registry._DEFAULT_BACKEND_NAME = None  # type: ignore[assignment]
    with pytest.raises(RuntimeError):
        registry.get_backend(None)


# --------------------------------------------------------------------------- #
# get_default_backend / set_default_backend                                   #
# --------------------------------------------------------------------------- #


def test_get_default_backend_returns_instance() -> None:
    b = registry.get_default_backend()
    assert isinstance(b, Backend)


def test_set_default_backend_changes_default() -> None:
    registry.register_backend("dummy", _DummyBackend)
    registry.set_default_backend("dummy")
    b = registry.get_default_backend()
    assert isinstance(b, _DummyBackend)


def test_set_default_unknown_raises() -> None:
    with pytest.raises(KeyError):
        registry.set_default_backend("nope")


# --------------------------------------------------------------------------- #
# list_available_backends                                                     #
# --------------------------------------------------------------------------- #


def test_list_available_backends_returns_sorted() -> None:
    registry.register_backend("dummy", _DummyBackend)
    registry.register_backend("aardvark", _DummyBackend)
    listed = registry.list_available_backends()
    assert listed == sorted(listed)
    assert "aardvark" in listed
    assert "dummy" in listed


def test_list_available_backends_returns_list() -> None:
    assert isinstance(registry.list_available_backends(), list)


# --------------------------------------------------------------------------- #
# Public re-exports                                                           #
# --------------------------------------------------------------------------- #


def test_public_api_exposed_from_package() -> None:
    import qtest.backends as pkg

    for symbol in (
        "Backend",
        "QiskitBackend",
        "get_backend",
        "get_default_backend",
        "set_default_backend",
        "register_backend",
        "list_available_backends",
    ):
        assert hasattr(pkg, symbol), f"qtest.backends should expose {symbol}"
