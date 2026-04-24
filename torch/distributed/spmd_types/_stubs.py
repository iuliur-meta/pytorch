"""
Stub implementations for when spmd_types is not installed.

Every function/class here mirrors the real spmd_types API but does nothing.
Type checking is simply disabled: assert_type returns the tensor unchanged,
typecheck() is a no-op context manager, type constants exist as sentinels.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any


class _StubType:
    """A named sentinel that stands in for an spmd_types type constant."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return self._name

    def __hash__(self) -> int:
        return hash(self._name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _StubType) and self._name == other._name


R = _StubType("R")
I = _StubType("I")
V = _StubType("V")
P = _StubType("P")


class _StubShard:
    """Stub for spmd_types S(dim)."""

    def __init__(self, dim: int = 0) -> None:
        self.dim = dim

    def __repr__(self) -> str:
        return f"S({self.dim})"

    def __hash__(self) -> int:
        return hash(("S", self.dim))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _StubShard) and self.dim == other.dim


S = _StubShard
SpmdShard = _StubShard


class SpmdTypeError(Exception):
    """Stub for spmd_types.types.SpmdTypeError."""


class MeshAxis:
    """Stub for spmd_types._mesh_axis.MeshAxis."""

    def __init__(self, key: Any = None) -> None:
        self._key = key

    @classmethod
    def of(cls, pg: Any, stride: Any = None) -> MeshAxis:
        return cls(key=id(pg))

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MeshAxis) and self._key == other._key

    def __repr__(self) -> str:
        return f"MeshAxis(stub, key={self._key})"


def assert_type(tensor: Any, type: Any = None, **kwargs: Any) -> Any:
    return tensor


def has_local_type(tensor: Any) -> bool:
    return False


def get_local_type(tensor: Any) -> dict:
    return {}


def set_local_type(tensor: Any, type: Any = None, **kwargs: Any) -> Any:
    return tensor


def get_partition_spec(*args: Any, **kwargs: Any) -> None:
    return None


def normalize_axis(*args: Any, **kwargs: Any) -> Any:
    return args[0] if args else None


def _reset() -> None:
    pass


@contextmanager
def typecheck(strict_mode: Any = None, local: Any = None):
    yield


@contextmanager
def set_current_mesh(axes: Any = None):
    yield


__all__ = [
    "R",
    "I",
    "V",
    "P",
    "S",
    "SpmdShard",
    "SpmdTypeError",
    "MeshAxis",
    "assert_type",
    "has_local_type",
    "get_local_type",
    "set_local_type",
    "typecheck",
    "set_current_mesh",
    "get_partition_spec",
    "normalize_axis",
    "_reset",
]
