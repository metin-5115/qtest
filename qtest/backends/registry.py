"""Module-level registry of available backends.

The registry stores a mapping ``{name: Backend subclass}`` and tracks which
name is the current default. The Qiskit backend is auto-registered on
import and set as the default.

Registration stores the *class* (a zero-argument-callable factory) so that
each call to :func:`get_backend` returns a fresh, stateless instance —
honouring the no-cross-call-state contract documented in
:class:`qtest.backends.base.Backend`.
"""

from __future__ import annotations

from qtest.backends.base import Backend

_REGISTRY: dict[str, type[Backend]] = {}
_DEFAULT_BACKEND_NAME: str | None = None


def register_backend(name: str, backend_class: type[Backend]) -> None:
    """Register *backend_class* under *name*.

    Subsequent registrations under the same name overwrite the previous
    entry.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    if not (isinstance(backend_class, type) and issubclass(backend_class, Backend)):
        raise TypeError(f"backend_class must be a subclass of Backend, got {backend_class!r}")
    _REGISTRY[name] = backend_class


def get_backend(name: str | None = None) -> Backend:
    """Instantiate and return a backend.

    If *name* is ``None``, the current default backend is used. Raises
    :class:`KeyError` if *name* is not registered, or :class:`RuntimeError`
    if no default is set.
    """
    if name is None:
        if _DEFAULT_BACKEND_NAME is None:
            raise RuntimeError(
                "No default backend is set. Use set_default_backend() or " "pass a name explicitly."
            )
        name = _DEFAULT_BACKEND_NAME
    if name not in _REGISTRY:
        raise KeyError(f"Unknown backend {name!r}. Available: {list_available_backends()}")
    return _REGISTRY[name]()


def get_default_backend() -> Backend:
    """Return a fresh instance of the current default backend."""
    return get_backend(None)


def set_default_backend(name: str) -> None:
    """Set *name* as the default backend. Must already be registered."""
    global _DEFAULT_BACKEND_NAME
    if name not in _REGISTRY:
        raise KeyError(
            f"Cannot set default to unknown backend {name!r}. "
            f"Available: {list_available_backends()}"
        )
    _DEFAULT_BACKEND_NAME = name


def list_available_backends() -> list[str]:
    """Return a sorted list of registered backend names."""
    return sorted(_REGISTRY)


def _auto_register_defaults() -> None:
    """Register the bundled Qiskit backend and make it the default.

    Importing :mod:`qtest.backends.qiskit_backend` does **not** import
    Qiskit (imports inside that module are lazy), so this call is safe
    even when Qiskit is not installed; an :class:`ImportError` is only
    raised when an execution method is actually called.
    """
    from qtest.backends.qiskit_backend import QiskitBackend

    register_backend("qiskit", QiskitBackend)
    global _DEFAULT_BACKEND_NAME
    if _DEFAULT_BACKEND_NAME is None:
        _DEFAULT_BACKEND_NAME = "qiskit"


_auto_register_defaults()
