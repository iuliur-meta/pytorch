"""
Re-export shim for spmd_types.

When the ``spmd_types`` package is installed, every public name from it
is re-exported here.  When it is *not* installed, lightweight stubs are
provided so that downstream code (FSDP, local_map) can import without
``try/except`` and type-checking features simply become no-ops.

Use ``is_available()`` to check at runtime whether the real package is
present.
"""

import importlib as _importlib


_HAS_SPMD_TYPES = _importlib.util.find_spec("spmd_types") is not None


def is_available() -> bool:
    """Return True if the real spmd_types package is installed."""
    return _HAS_SPMD_TYPES


if _HAS_SPMD_TYPES:
    _mod = _importlib.import_module("spmd_types")
    _public = [name for name in dir(_mod) if not name.startswith("_")]
    globals().update({name: getattr(_mod, name) for name in _public})
    __all__ = _public + ["is_available"]  # noqa: PLE0605

    from spmd_types._checker import (  # pyrefly: ignore
        get_partition_spec,
        no_typecheck,
        typecheck,
    )
    from spmd_types._mesh import set_current_mesh  # pyrefly: ignore
    from spmd_types._mesh_axis import _reset, MeshAxis  # pyrefly: ignore
    from spmd_types._type_attr import get_local_type, set_local_type  # pyrefly: ignore
    from spmd_types.runtime import assert_type, has_local_type  # pyrefly: ignore
    from spmd_types.types import (  # pyrefly: ignore
        I,
        normalize_axis,
        P,
        R,
        S,
        Shard as SpmdShard,
        SpmdTypeError,
        V,
    )
else:
    from torch.distributed.spmd_types._stubs import *  # noqa: F403
    from torch.distributed.spmd_types._stubs import (
        _reset,
        assert_type,
        get_local_type,
        get_partition_spec,
        has_local_type,
        MeshAxis,
        no_typecheck,
        normalize_axis,
        set_current_mesh,
        set_local_type,
        SpmdShard,
        SpmdTypeError,
        typecheck,
    )
