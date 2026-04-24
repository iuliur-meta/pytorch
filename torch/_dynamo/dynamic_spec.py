"""Dynamic shape specification types for ``torch.compile`` and ``torch.export``.

Provides :class:`IntSpec` for fine-grained control over whether an integer
(dimension size or scalar argument) is treated as static, backed, or unbacked
during compilation.

Backed vs. unbacked
-------------------
``torch.compile`` provides two kinds of dynamic shapes: ``backed`` and
``unbacked``. ``torch.compile`` guards on ``backed`` dynamic shapes and does
not provide a guarantee that no guards will be added to them. User code,
dynamo, inductor, and autograd all can add guards when tracing through
branching, e.g. ``if x.size() > 10``. Moreover, for 0/1 specializations,
backed symbols are specialized unconditionally to ``0``, ``1``, or ``>=2``
even without encountering a branching on those ranges.

On the contrary, ``unbacked`` dynamic shapes are guaranteed not to be guarded
on and are not 0/1 specialized. However, there is a possibility of throwing a
data-dependent error when a branch that requires their value is encountered
and no explicit unbacked handling is defined. The framework is converging to
a state where it won't throw DDE but rather pick general paths. One downside
of using unbacked is missed optimization opportunities due to either perf
bugs or picking general paths, or using a fixed non-example input-based hint.
An example of picking general paths is assuming input not contiguous in
functions called ``contiguous()`` and ``reshape()`` when it cannot be
symbolically proven, with a change of introducing a clone.

For more info see
https://dev-discuss.pytorch.org/t/backed-to-unbacked-from-guardable-to-guardless-shapes-in-pytorch/3333.
"""

import enum
from collections.abc import Iterator
from contextvars import ContextVar
from typing import Any, ClassVar, NoReturn


__all__ = ["IntSpecType", "IntSpec", "TensorSpec", "ModelSpec"]


class IntSpecType(enum.Enum):
    """How an integer should be treated during compilation.

    STATIC
        Compile-time constant; recompiles if the value changes.
    BACKED
        Symbolic with a guarding hint. Guards and 0/1 specialization are
        permitted.
    UNBACKED
        Symbolic with an optimized hint. Not guarded on, not 0/1
        specialized; branching on the value may raise a data-dependent
        error.
    """

    STATIC = "static"
    BACKED = "backed"
    UNBACKED = "unbacked"


class IntSpec:
    """Shape specification for a single integer (dimension size or scalar arg).

    Constructed via one of the three mode-specific classmethod factories —
    :meth:`static`, :meth:`backed`, :meth:`unbacked` — or by calling the
    constructor directly with an explicit :class:`IntSpecType`. The mode must
    be supplied at construction time; there is no valid "unspecified" mode.

    An :class:`IntSpec` is **immutable** once constructed. All fields —
    ``name``, ``type``, ``min``, ``max``, ``value``, ``guarding_hint``,
    ``optimization_hint`` — are fixed at construction time. Attempting to
    reassign any of them (including the private ``_type``, ``_value``, etc.
    backing slots) raises :class:`AttributeError`. To produce a spec with
    different values, construct a new instance or use a fluent setter (e.g.
    ``spec.guarding_hint(64)``), which returns a fresh :class:`IntSpec`.

    **Why immutable for every field, not just `type`.** From a pure JIT
    correctness standpoint only ``type`` (mode) is a hard semantic commitment
    — ``value`` / ``guarding_hint`` / ``optimization_hint`` are tuning knobs
    whose mutation would only shift specialization or autotuning decisions.
    But AOT flows (``torch.export``, ``AOTInductor``) snapshot the spec's
    fields into the compiled artifact at capture time: mutating a hint after
    export would silently desync the live spec from the sealed artifact
    (compiled code says 32, live spec says 64, no error signal). Compilation
    caches have the same requirement — cache keys must be content-addressed,
    not reference-addressed, which is only safe if the spec's identity can't
    drift. Rather than partition the invariant by consumer, the class holds
    the stronger one uniformly; incremental updates go through
    ``_replace`` / fluent setters, which yield fresh instances.

    Example::

        IntSpec.static("x", value=10)
        IntSpec.backed("batch", min=1, max=64, guarding_hint=32)
        IntSpec.unbacked("seq", min=1, max=2048, optimization_hint=512)

        # Anonymous form (scalar-int use, no name):
        IntSpec.backed()
        IntSpec.backed(guarding_hint=32)

        # Direct constructor (name required, may be None):
        IntSpec("x", IntSpecType.STATIC, value=10)
    """

    # Slot annotations. Pyrefly / mypy can't see the backing slots because
    # ``__init__`` populates them via ``object.__setattr__(self, "<name>", ...)``
    # with string literals to bypass our own ``__setattr__`` override. These
    # annotations make the slot types visible to static checkers without
    # creating class-level values.
    _name: str | None
    _type: IntSpecType
    _min: int | None
    _max: int | None
    _value: int | None
    _guarding_hint: int | None
    _optimization_hint: int | None

    __slots__ = (
        "_name",
        "_type",
        "_min",
        "_max",
        "_value",
        "_guarding_hint",
        "_optimization_hint",
    )

    def __init__(
        self,
        name: str | None,
        type: IntSpecType,
        *,
        min: int | None = None,
        max: int | None = None,
        value: int | None = None,
        guarding_hint: int | None = None,
        optimization_hint: int | None = None,
    ) -> None:
        if not isinstance(type, IntSpecType):
            raise TypeError(f"IntSpec.type must be an IntSpecType, got {type!r}")
        if name is not None and not isinstance(name, str):
            raise TypeError(
                f"IntSpec.name must be str or None, got "
                f"{name.__class__.__name__}; if you meant to pass a "
                f"value/hint, use a keyword argument "
                f"(e.g. IntSpec.static(value=10))"
            )
        for field_name, field_val in (
            ("min", min),
            ("max", max),
            ("value", value),
            ("guarding_hint", guarding_hint),
            ("optimization_hint", optimization_hint),
        ):
            if field_val is not None and (
                not isinstance(field_val, int) or isinstance(field_val, bool)
            ):
                raise TypeError(
                    f"IntSpec.{field_name} must be int or None, got "
                    f"{field_val.__class__.__name__}"
                )
        # Bypass our own __setattr__ (which blocks all writes post-construction)
        # by going through object.__setattr__ during initialization.
        setattr_ = object.__setattr__
        setattr_(self, "_name", name)
        setattr_(self, "_type", type)
        setattr_(self, "_min", min)
        setattr_(self, "_max", max)
        setattr_(self, "_value", value)
        setattr_(self, "_guarding_hint", guarding_hint)
        setattr_(self, "_optimization_hint", optimization_hint)
        self._validate()

    def __setattr__(self, key: str, value: Any) -> NoReturn:
        raise AttributeError(f"IntSpec is immutable; cannot set attribute {key!r}")

    def __delattr__(self, key: str) -> NoReturn:
        raise AttributeError(f"IntSpec is immutable; cannot delete attribute {key!r}")

    # -- validation --------------------------------------------------------
    #
    # Per-mode field rules are expressed as two tables instead of a branching
    # tree: ``_ALLOWED_FIELDS`` lists which non-identity fields each mode
    # accepts, and ``_FIELD_REJECT_MESSAGE`` holds the user-facing message
    # for every field that can be rejected. ``_validate`` walks each set
    # field and rejects any that isn't allowed for the mode. Cross-field
    # checks (like ``min <= max``) are spelled out below the loop.

    _ALLOWED_FIELDS: ClassVar[dict[IntSpecType, frozenset[str]]] = {
        IntSpecType.STATIC: frozenset({"value"}),
        IntSpecType.BACKED: frozenset({"min", "max", "guarding_hint"}),
        IntSpecType.UNBACKED: frozenset({"min", "max", "optimization_hint"}),
    }

    _FIELD_REJECT_MESSAGE: ClassVar[dict[str, str]] = {
        "value": "value is only valid for STATIC IntSpec",
        "guarding_hint": "guarding_hint is only valid for BACKED IntSpec",
        "optimization_hint": "optimization_hint is only valid for UNBACKED IntSpec",
        "min": "min/max are only valid for BACKED/UNBACKED IntSpec, not STATIC",
        "max": "min/max are only valid for BACKED/UNBACKED IntSpec, not STATIC",
    }

    def _validate(self) -> None:
        allowed = IntSpec._ALLOWED_FIELDS[self._type]
        fields = {
            "min": self._min,
            "max": self._max,
            "value": self._value,
            "guarding_hint": self._guarding_hint,
            "optimization_hint": self._optimization_hint,
        }
        for name, val in fields.items():
            if val is None or name in allowed:
                continue
            raise ValueError(IntSpec._FIELD_REJECT_MESSAGE[name])
        if self._min is not None and self._max is not None and self._min > self._max:
            raise ValueError(
                f"min must be <= max, got min={self._min}, max={self._max}"
            )

    # -- factories ---------------------------------------------------------
    #
    # ``name`` is the only positional argument; all other fields are
    # keyword-only. Passing non-``str`` (e.g. ``IntSpec.static(10)``) is
    # rejected at ``__init__`` with a hint to use the kwarg form.

    @classmethod
    def static(cls, name: str | None = None, *, value: int | None = None) -> "IntSpec":
        """Construct a STATIC :class:`IntSpec`.

        ``value`` pins a concrete size; if ``None`` the value is taken from
        the example input at compile time.
        """
        return cls(name, type=IntSpecType.STATIC, value=value)

    @classmethod
    def backed(
        cls,
        name: str | None = None,
        *,
        min: int | None = None,
        max: int | None = None,
        guarding_hint: int | None = None,
    ) -> "IntSpec":
        """Construct a BACKED :class:`IntSpec`.

        ``guarding_hint`` is the concrete value the symbolic shape
        environment substitutes when a hint is needed for reasoning or
        codegen.
        """
        return cls(
            name,
            type=IntSpecType.BACKED,
            min=min,
            max=max,
            guarding_hint=guarding_hint,
        )

    @classmethod
    def unbacked(
        cls,
        name: str | None = None,
        *,
        min: int | None = None,
        max: int | None = None,
        optimization_hint: int | None = None,
    ) -> "IntSpec":
        """Construct an UNBACKED :class:`IntSpec`.

        ``optimization_hint`` is used by downstream codegen (e.g. inductor
        autotuning) only; it never participates in symbolic reasoning.
        """
        return cls(
            name,
            type=IntSpecType.UNBACKED,
            min=min,
            max=max,
            optimization_hint=optimization_hint,
        )

    # -- identity (read) ---------------------------------------------------

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def type(self) -> IntSpecType:
        return self._type

    # -- fluent setters ----------------------------------------------------
    #
    # Each returns a new :class:`IntSpec` with the given field replaced; the
    # receiver is unchanged. Per-mode validity (e.g. ``guarding_hint`` only
    # on BACKED) is enforced by the constructor, so e.g.
    # ``IntSpec.static("x").guarding_hint(10)`` raises ``ValueError``.
    #
    # Users who want to inspect a spec's field values should use ``repr()``;
    # the backing-slot reads (``_min``/``_max``/etc.) are private to the
    # module and consumed by the dynamo integration (and privileged tests
    # that verify the integration landed a value in the right slot).

    # Private: only the five fluent setters below call this, each passing
    # one of the five replaceable data fields by name. ``name`` and ``type``
    # stay pinned from the original instance — changing them would not be
    # a "replace" operation. External use is unsupported — users should
    # chain fluent setters directly.
    def _replace(self, **overrides: int | None) -> "IntSpec":
        return IntSpec(
            self._name,
            self._type,
            min=overrides.get("min", self._min),
            max=overrides.get("max", self._max),
            value=overrides.get("value", self._value),
            guarding_hint=overrides.get("guarding_hint", self._guarding_hint),
            optimization_hint=overrides.get(
                "optimization_hint", self._optimization_hint
            ),
        )

    def min(self, value: int) -> "IntSpec":
        return self._replace(min=value)

    def max(self, value: int) -> "IntSpec":
        return self._replace(max=value)

    def value(self, value: int) -> "IntSpec":
        return self._replace(value=value)

    def guarding_hint(self, value: int) -> "IntSpec":
        return self._replace(guarding_hint=value)

    def optimization_hint(self, value: int) -> "IntSpec":
        return self._replace(optimization_hint=value)

    # -- dunder ------------------------------------------------------------

    def __repr__(self) -> str:
        parts: list[str] = []
        if self.name is not None:
            parts.append(f"name={self.name!r}")
        parts.append(f"type={self._type.name}")
        if self._value is not None:
            parts.append(f"value={self._value}")
        if self._min is not None:
            parts.append(f"min={self._min}")
        if self._max is not None:
            parts.append(f"max={self._max}")
        if self._guarding_hint is not None:
            parts.append(f"guarding_hint={self._guarding_hint}")
        if self._optimization_hint is not None:
            parts.append(f"optimization_hint={self._optimization_hint}")
        return f"IntSpec({', '.join(parts)})"


class TensorSpec:
    """Per-dimension shape specification for a tensor.

    A list-like container of ``IntSpec | None`` with length equal to the
    tensor's rank. ``None`` entries inherit the default dynamism policy from
    the compile context.

    Example::

        ts = TensorSpec(3)
        ts.set(0, IntSpec.backed("batch", min=1, max=64))
        # dims 1 and 2 are None -> inherit context default
    """

    def __init__(self, rank: int) -> None:
        if rank < 0:
            raise ValueError(f"rank must be non-negative, got {rank}")
        self._rank = rank
        self._specs: list[IntSpec | None] = [None] * rank

    @classmethod
    def from_list(cls, specs: list[IntSpec | None]) -> "TensorSpec":
        """Construct from an existing list of specs."""
        ts = cls(len(specs))
        ts._specs = list(specs)
        return ts

    @property
    def rank(self) -> int:
        return self._rank

    def set(self, index: int, spec: IntSpec) -> "TensorSpec":
        """Set the spec at ``index`` and return ``self`` for chaining."""
        self._specs[index] = spec
        return self

    def __getitem__(self, index: int) -> IntSpec | None:
        return self._specs[index]

    def __setitem__(self, index: int, spec: IntSpec | None) -> None:
        self._specs[index] = spec

    def __len__(self) -> int:
        return self._rank

    def __iter__(self) -> Iterator[IntSpec | None]:
        return iter(self._specs)

    def __repr__(self) -> str:
        specified = [
            f"{i}: {spec!r}" for i, spec in enumerate(self._specs) if spec is not None
        ]
        return f"TensorSpec(rank={self._rank}, {{{', '.join(specified)}}})"

    # No ``__eq__`` / ``__hash__``: matches :class:`IntSpec`'s design — specs
    # are immutable compile-time inputs compared via ``repr()`` when needed.


class ModelSpec:
    """Top-level dynamic-shape specification for a whole compiled model.

    A dict-like container mapping argument names (as they appear in the
    compiled function's signature) to per-argument specs. Per-argument spec
    can be:

    - :class:`TensorSpec` — per-dimension spec for a tensor argument.
    - :class:`IntSpec` — spec for a scalar integer argument.
    - ``dict[int, IntSpec | None]`` — sparse per-dim spec.
    - ``list[IntSpec | None]`` / ``tuple[IntSpec | None, ...]`` — positional
      per-dim spec.
    - ``None`` — inherit the compile-context default for that argument.

    Example::

        ModelSpec(
            {
                "x": TensorSpec(2).set(0, IntSpec.backed("batch")),
                "batch_size": IntSpec.backed("batch"),
            }
        )
    """

    def __init__(self, specs: dict[str, Any] | None = None) -> None:
        self._specs: dict[str, Any] = dict(specs) if specs else {}

    def set(self, name: str, spec: Any) -> "ModelSpec":
        """Assign *spec* to the argument *name*. Returns ``self`` for chaining."""
        self._specs[name] = spec
        return self

    def __getitem__(self, name: str) -> Any:
        return self._specs[name]

    def __setitem__(self, name: str, spec: Any) -> None:
        self._specs[name] = spec

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __iter__(self) -> Iterator[str]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def items(self) -> Any:
        return self._specs.items()

    def get(self, name: str, default: Any = None) -> Any:
        return self._specs.get(name, default)

    def __repr__(self) -> str:
        return f"ModelSpec({self._specs!r})"

    # No ``__eq__`` / ``__hash__``: matches :class:`IntSpec` / :class:`TensorSpec`.


# ContextVar carrying the dynamic_shapes spec for the currently-running
# ``torch.compile``'d function. Set by :func:`_apply_dynamic_shapes` on each
# call; read by :func:`get_active_spec_for_dim` from inside the Dynamo
# variable builder during input wrapping. No tensor monkey-patching; no
# pre-installed guards — the spec directly selects ``DimDynamic`` in
# ``_automatic_dynamic``.
_active_dynamic_shapes: ContextVar[dict[str, Any] | None] = ContextVar(
    "_dynamo_active_dynamic_shapes", default=None
)


def _resolve_dim_spec(arg_spec: Any, dim: int) -> "IntSpec | None":
    """Extract the :class:`IntSpec` for *dim* from a per-argument spec.

    Supports the four forms accepted in ``dynamic_shapes``: ``TensorSpec``,
    ``dict[int, IntSpec]``, ``list``/``tuple`` of IntSpec-or-None, or an
    :class:`IntSpec` directly (for scalar-int arguments; ``dim`` ignored).
    """
    if isinstance(arg_spec, IntSpec):
        return arg_spec
    if isinstance(arg_spec, TensorSpec):
        return arg_spec[dim] if 0 <= dim < len(arg_spec) else None
    if isinstance(arg_spec, dict):
        return arg_spec.get(dim)
    if isinstance(arg_spec, (list, tuple)):
        return arg_spec[dim] if 0 <= dim < len(arg_spec) else None
    return None


def get_active_spec_for_arg(arg_name: str) -> Any:
    """Return the spec associated with *arg_name* in the active
    ``dynamic_shapes``, or ``None`` if no spec is active or the arg is not
    listed."""
    spec_dict = _active_dynamic_shapes.get()
    if spec_dict is None:
        return None
    if isinstance(spec_dict, ModelSpec):
        return spec_dict.get(arg_name)
    return spec_dict.get(arg_name)


def get_active_spec_for_dim(arg_name: str, dim: int) -> "IntSpec | None":
    """Return the :class:`IntSpec` for *dim* of argument *arg_name* in the
    active ``dynamic_shapes``, or ``None``."""
    arg_spec = get_active_spec_for_arg(arg_name)
    if arg_spec is None:
        return None
    return _resolve_dim_spec(arg_spec, dim)
