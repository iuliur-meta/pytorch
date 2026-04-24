# Owner(s): ["module: dynamo"]

import functools
import inspect

import torch
import torch._dynamo
import torch._dynamo.testing
import torch.fx.experimental._config as _fx_experimental_config
from torch._dynamo.decorators import mark_static, mark_unbacked, maybe_mark_dynamic
from torch._dynamo.dynamic_spec import IntSpec, IntSpecType
from torch._dynamo.test_case import TestCase
from torch._dynamo.testing import EagerAndRecordGraphs
from torch.fx.experimental.symbolic_shapes import (
    free_unbacked_symbols,
    GuardOnDataDependentSymNode,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
)


def _tensor_placeholder_shape(gm):
    """Return the shape of the first tensor-typed placeholder in ``gm``."""
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            ev = node.meta.get("example_value")
            if isinstance(ev, torch.Tensor):
                return ev.shape
    raise AssertionError("no tensor placeholder found")


# Test-only scaffolding. Mimics what the PR 3 integration will do for real via
# the compile context: apply per-dim IntSpec to tensors through ``mark_*`` on
# each call. Kept here (and not in production) because the approach installs
# guards eagerly and doesn't cover scalar int inputs — both issues that the
# real integration resolves.
def _apply_intspec_to_tensor(tensor, shape_spec):
    if isinstance(shape_spec, dict):
        items = shape_spec.items()
    elif isinstance(shape_spec, (list, tuple)):
        items = enumerate(shape_spec)
    else:
        return
    for idx, spec in items:
        if spec is None:
            continue
        if not isinstance(spec, IntSpec):
            raise TypeError(
                f"Expected IntSpec or None in dynamic_shapes, got {type(spec).__name__}"
            )
        if spec.type is IntSpecType.STATIC:
            mark_static(tensor, idx)
        elif spec.type is IntSpecType.BACKED:
            maybe_mark_dynamic(tensor, idx)
        elif spec.type is IntSpecType.UNBACKED:
            mark_unbacked(tensor, idx)


def _compile_with_dynamic_shapes(fn, dynamic_shapes, **compile_kwargs):
    """Compile ``fn`` and wrap it so ``dynamic_shapes`` is applied per call.

    Test-only replacement for the (now-removed) ``torch.compile(dynamic_shapes=...)``
    scaffolding. The outer wrapper is ``torch._dynamo.disable``'d so dynamo
    does not trace the tensor-marking logic; the inner ``compiled()`` call
    re-enters dynamo.
    """
    compiled = torch.compile(fn, **compile_kwargs)
    sig = inspect.signature(fn.forward if isinstance(fn, torch.nn.Module) else fn)

    @torch._dynamo.disable
    @functools.wraps(
        compiled if not isinstance(compiled, torch.nn.Module) else compiled.forward
    )
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for name, shape_spec in dynamic_shapes.items():
            if name in bound.arguments:
                arg = bound.arguments[name]
                if isinstance(arg, torch.Tensor):
                    _apply_intspec_to_tensor(arg, shape_spec)
        return compiled(*bound.args, **bound.kwargs)

    return wrapper


class TestIntSpecConstruction(TestCase):
    """Construction via the classmethod factories.

    There are deliberately no public read properties for ``min`` / ``max`` /
    ``value`` / ``guarding_hint`` / ``optimization_hint`` — those names are
    fluent setters that return a new :class:`IntSpec`. Users inspect field
    values via ``repr()``; tests in this file read private slots directly
    (``s._min`` etc.) because they're privileged — asserting "the constructor
    landed this value in this slot" is implementation-level verification, not
    a user-facing API.
    """

    def test_static(self):
        s = IntSpec.static("x", value=10)
        self.assertEqual(s.name, "x")
        self.assertEqual(s.type, IntSpecType.STATIC)
        self.assertEqual(s._value, 10)
        self.assertIn("value=10", repr(s))

    def test_static_no_value(self):
        s = IntSpec.static()
        self.assertEqual(s.type, IntSpecType.STATIC)
        self.assertIsNone(s._value)
        self.assertNotIn("value=", repr(s))

    def test_backed(self):
        s = IntSpec.backed("batch", min=1, max=64, guarding_hint=32)
        self.assertEqual(s.name, "batch")
        self.assertEqual(s.type, IntSpecType.BACKED)
        self.assertEqual(s._min, 1)
        self.assertEqual(s._max, 64)
        self.assertEqual(s._guarding_hint, 32)

    def test_unbacked(self):
        s = IntSpec.unbacked("seq", min=1, max=2048, optimization_hint=512)
        self.assertEqual(s.type, IntSpecType.UNBACKED)
        self.assertEqual(s._min, 1)
        self.assertEqual(s._max, 2048)
        self.assertEqual(s._optimization_hint, 512)

    def test_type_required_on_init(self):
        with self.assertRaises(TypeError) as cm:
            IntSpec("x")  # no type kwarg
        # Python-generated message; format pinned for 3.10+.
        self.assertEqual(
            str(cm.exception),
            "IntSpec.__init__() missing 1 required positional argument: 'type'",
        )

    def test_type_not_none(self):
        with self.assertRaises(TypeError) as cm:
            IntSpec("x", type=None)  # type: ignore[arg-type]
        self.assertEqual(
            str(cm.exception),
            "IntSpec.type must be an IntSpecType, got None",
        )

    def test_type_as_positional_arg(self):
        s = IntSpec("batch", IntSpecType.BACKED, min=1, max=64)
        self.assertEqual(s.name, "batch")
        self.assertEqual(s.type, IntSpecType.BACKED)
        self.assertEqual(s._min, 1)
        self.assertEqual(s._max, 64)

    def test_static_with_positional_int_rejected(self):
        # ``IntSpec.static(10)`` would silently bind 10 to ``name``. Must
        # fail with a clear redirect to the kwarg form.
        with self.assertRaises(TypeError) as cm:
            IntSpec.static(10)  # type: ignore[arg-type]
        self.assertEqual(
            str(cm.exception),
            "IntSpec.name must be str or None, got int; "
            "if you meant to pass a value/hint, use a keyword argument "
            "(e.g. IntSpec.static(value=10))",
        )

    def test_backed_with_positional_int_rejected(self):
        with self.assertRaises(TypeError) as cm:
            IntSpec.backed(5)  # type: ignore[arg-type]
        self.assertEqual(
            str(cm.exception),
            "IntSpec.name must be str or None, got int; "
            "if you meant to pass a value/hint, use a keyword argument "
            "(e.g. IntSpec.static(value=10))",
        )

    def test_unbacked_with_positional_int_rejected(self):
        with self.assertRaises(TypeError) as cm:
            IntSpec.unbacked(5)  # type: ignore[arg-type]
        self.assertEqual(
            str(cm.exception),
            "IntSpec.name must be str or None, got int; "
            "if you meant to pass a value/hint, use a keyword argument "
            "(e.g. IntSpec.static(value=10))",
        )

    def test_name_wrong_type_on_init(self):
        with self.assertRaises(TypeError) as cm:
            IntSpec(10, IntSpecType.STATIC)  # type: ignore[arg-type]
        self.assertEqual(
            str(cm.exception),
            "IntSpec.name must be str or None, got int; "
            "if you meant to pass a value/hint, use a keyword argument "
            "(e.g. IntSpec.static(value=10))",
        )


class TestIntSpecImmutable(TestCase):
    """Once constructed, every field is fixed; no write path (public name,
    private slot, new attribute) is allowed. Fluent setters return fresh
    instances; the receiver is unchanged — covered in TestIntSpecFluent."""

    def test_type_is_read_only(self):
        s = IntSpec.static("x", value=10)
        with self.assertRaises(AttributeError) as cm:
            s.type = IntSpecType.BACKED  # type: ignore[misc]
        self.assertEqual(
            str(cm.exception),
            "IntSpec is immutable; cannot set attribute 'type'",
        )

    def test_no_fluent_type_reset(self):
        # IntSpec has no instance method that reassigns type. The mode-named
        # factories are classmethods: calling one "on an instance" returns a
        # fresh IntSpec and does not mutate the original.
        s = IntSpec.static("x")
        new = IntSpec.backed("x")
        self.assertIs(s.type, IntSpecType.STATIC)
        self.assertIs(new.type, IntSpecType.BACKED)
        self.assertIsNot(s, new)

    def test_name_is_read_only(self):
        s = IntSpec.static("x")
        with self.assertRaises(AttributeError) as cm:
            s.name = "y"  # type: ignore[misc]
        self.assertEqual(
            str(cm.exception),
            "IntSpec is immutable; cannot set attribute 'name'",
        )

    def test_cannot_shadow_fluent_method_on_instance(self):
        # Assigning to a fluent-method name like ``value`` / ``min`` /
        # ``guarding_hint`` on an instance must still raise — __slots__ has
        # no matching slot and __setattr__ rejects all writes.
        s = IntSpec.static("x", value=10)
        for attr in (
            "value",
            "min",
            "max",
            "guarding_hint",
            "optimization_hint",
        ):
            with self.assertRaises(AttributeError) as cm:
                setattr(s, attr, 20)
            self.assertEqual(
                str(cm.exception),
                f"IntSpec is immutable; cannot set attribute {attr!r}",
            )

    def test_cannot_reassign_type_via_private_slot(self):
        # Dedicated test for the private-slot backdoor on ``_type``. Without
        # our ``__setattr__`` override, a user could swap the mode after
        # construction — precisely the footgun laithsakka pushed to close.
        s = IntSpec.backed("x")
        with self.assertRaises(AttributeError) as cm:
            s._type = IntSpecType.STATIC
        self.assertEqual(
            str(cm.exception),
            "IntSpec is immutable; cannot set attribute '_type'",
        )
        # Confirm the mode actually did not change.
        self.assertIs(s.type, IntSpecType.BACKED)

    def test_private_slots_cannot_be_reassigned(self):
        # The private backing slots are also locked — no "backdoor" mutation.
        s = IntSpec.backed("x", min=1, max=64, guarding_hint=32)
        for attr, new_val in [
            ("_type", IntSpecType.STATIC),
            ("_name", "y"),
            ("_min", 0),
            ("_max", 128),
            ("_value", 10),
            ("_guarding_hint", 64),
            ("_optimization_hint", 100),
        ]:
            with self.assertRaises(AttributeError) as cm:
                setattr(s, attr, new_val)
            self.assertEqual(
                str(cm.exception),
                f"IntSpec is immutable; cannot set attribute {attr!r}",
            )

    def test_cannot_add_new_attribute(self):
        s = IntSpec.static("x")
        with self.assertRaises(AttributeError) as cm:
            s.brand_new_field = 1  # type: ignore[attr-defined]
        self.assertEqual(
            str(cm.exception),
            "IntSpec is immutable; cannot set attribute 'brand_new_field'",
        )

    def test_cannot_delete_attribute(self):
        s = IntSpec.backed("x", guarding_hint=32)
        with self.assertRaises(AttributeError) as cm:
            del s._guarding_hint
        self.assertEqual(
            str(cm.exception),
            "IntSpec is immutable; cannot delete attribute '_guarding_hint'",
        )


class TestIntSpecFluent(TestCase):
    """Fluent setters (``min`` / ``max`` / ``value`` / ``guarding_hint`` /
    ``optimization_hint``) return a new immutable :class:`IntSpec` with one
    field replaced. Each chain is equivalent to the kwargs-only factory form.
    """

    def test_unbacked_chain_matches_kwargs_form(self):
        fluent = IntSpec.unbacked("seq").min(1).max(2048).optimization_hint(512)
        kwargs = IntSpec.unbacked("seq", min=1, max=2048, optimization_hint=512)
        self.assertEqual(repr(fluent), repr(kwargs))
        self.assertEqual(fluent._min, kwargs._min)
        self.assertEqual(fluent._max, kwargs._max)
        self.assertEqual(fluent._optimization_hint, kwargs._optimization_hint)

    def test_backed_chain_matches_kwargs_form(self):
        fluent = IntSpec.backed("batch").min(1).max(64).guarding_hint(32)
        kwargs = IntSpec.backed("batch", min=1, max=64, guarding_hint=32)
        self.assertEqual(repr(fluent), repr(kwargs))

    def test_static_value_chain_matches_kwargs_form(self):
        fluent = IntSpec.static("x").value(10)
        kwargs = IntSpec.static("x", value=10)
        self.assertEqual(repr(fluent), repr(kwargs))
        self.assertEqual(fluent._value, 10)

    def test_chain_returns_new_instance_receiver_unchanged(self):
        base = IntSpec.unbacked("seq")
        chained = base.min(1).max(2048)
        self.assertIsNot(base, chained)
        # Receiver didn't mutate — its slots are still the factory defaults.
        self.assertIsNone(base._min)
        self.assertIsNone(base._max)
        # And the chain's intermediate steps returned distinct objects too.
        step1 = base.min(1)
        step2 = step1.max(2048)
        self.assertIsNot(base, step1)
        self.assertIsNot(step1, step2)

    def test_fluent_preserves_existing_fields(self):
        # Chaining ``.max(2048)`` on a spec that already has a
        # ``guarding_hint`` must retain the hint.
        s = IntSpec.backed("batch", min=1, guarding_hint=32).max(64)
        self.assertEqual(s._min, 1)
        self.assertEqual(s._max, 64)
        self.assertEqual(s._guarding_hint, 32)


class TestIntSpecValidation(TestCase):
    """Cross-field validation that isn't per-mode (e.g. ``min > max``).

    Per-mode field rejection rules live in :class:`TestIntSpecRejectionRules`.
    """

    def test_backed_min_greater_than_max(self):
        with self.assertRaises(ValueError) as cm:
            IntSpec.backed("x", min=100, max=1)
        self.assertEqual(
            str(cm.exception),
            "min must be <= max, got min=100, max=1",
        )

    def test_unbacked_min_greater_than_max(self):
        with self.assertRaises(ValueError) as cm:
            IntSpec.unbacked("x", min=100, max=1)
        self.assertEqual(
            str(cm.exception),
            "min must be <= max, got min=100, max=1",
        )


@instantiate_parametrized_tests
class TestIntSpecRejectionRules(TestCase):
    """Per-mode field-rejection rules, exercised via both entry points.

    Each rule is the same ``IntSpec._validate`` check reached from two
    user-visible paths:

    - ``init``: direct kwargs at the raw constructor, e.g.
      ``IntSpec("x", IntSpecType.STATIC, guarding_hint=10)``.
    - ``fluent``: fluent setter chained off a factory, e.g.
      ``IntSpec.static("x").guarding_hint(10)``.

    Both paths must produce the same error message because both go through
    the constructor (the fluent setter's ``_replace`` helper rebuilds a
    fresh :class:`IntSpec`). Parametrizing the entry-point axis keeps a
    single source of truth per rule.
    """

    @parametrize("entry", ["init", "fluent"])
    @parametrize("mode", [IntSpecType.BACKED, IntSpecType.UNBACKED])
    def test_value_rejected_on_non_static(self, mode, entry):
        if entry == "init":
            ctor = lambda: IntSpec("x", mode, value=42)  # noqa: E731
        else:
            factory = IntSpec.backed if mode is IntSpecType.BACKED else IntSpec.unbacked
            ctor = lambda: factory("x").value(42)  # noqa: E731
        with self.assertRaises(ValueError) as cm:
            ctor()
        self.assertEqual(str(cm.exception), "value is only valid for STATIC IntSpec")

    @parametrize("entry", ["init", "fluent"])
    @parametrize("mode", [IntSpecType.STATIC, IntSpecType.UNBACKED])
    def test_guarding_hint_rejected_on_non_backed(self, mode, entry):
        if entry == "init":
            ctor = lambda: IntSpec("x", mode, guarding_hint=10)  # noqa: E731
        else:
            factory = IntSpec.static if mode is IntSpecType.STATIC else IntSpec.unbacked
            ctor = lambda: factory("x").guarding_hint(10)  # noqa: E731
        with self.assertRaises(ValueError) as cm:
            ctor()
        self.assertEqual(
            str(cm.exception), "guarding_hint is only valid for BACKED IntSpec"
        )

    @parametrize("entry", ["init", "fluent"])
    @parametrize("mode", [IntSpecType.STATIC, IntSpecType.BACKED])
    def test_optimization_hint_rejected_on_non_unbacked(self, mode, entry):
        if entry == "init":
            ctor = lambda: IntSpec("x", mode, optimization_hint=10)  # noqa: E731
        else:
            factory = IntSpec.static if mode is IntSpecType.STATIC else IntSpec.backed
            ctor = lambda: factory("x").optimization_hint(10)  # noqa: E731
        with self.assertRaises(ValueError) as cm:
            ctor()
        self.assertEqual(
            str(cm.exception),
            "optimization_hint is only valid for UNBACKED IntSpec",
        )

    @parametrize("entry", ["init", "fluent"])
    @parametrize("field", ["min", "max"])
    def test_min_max_rejected_on_static(self, field, entry):
        if entry == "init":
            ctor = lambda: IntSpec("x", IntSpecType.STATIC, **{field: 1})  # noqa: E731
        else:
            ctor = lambda: getattr(IntSpec.static("x"), field)(1)  # noqa: E731
        with self.assertRaises(ValueError) as cm:
            ctor()
        self.assertEqual(
            str(cm.exception),
            "min/max are only valid for BACKED/UNBACKED IntSpec, not STATIC",
        )

    # Each case: (field, bad_value, factory_name, expected_type_name).
    # ``factory_name`` is the factory whose mode allows the field — we
    # exercise the type check on the valid mode since mode-rejection is
    # already covered by the four tests above.
    _TYPE_CASES = [
        ("value", "10", "static", "str"),
        ("min", 1.5, "backed", "float"),
        ("max", "64", "backed", "str"),
        ("guarding_hint", True, "backed", "bool"),
        ("optimization_hint", 1.0, "unbacked", "float"),
    ]

    @parametrize("entry", ["init", "fluent"])
    @parametrize("field,bad,factory_name,type_name", _TYPE_CASES)
    def test_field_type_rejected(self, field, bad, factory_name, type_name, entry):
        factory = getattr(IntSpec, factory_name)
        if entry == "init":
            ctor = lambda: factory("x", **{field: bad})  # noqa: E731
        else:
            ctor = lambda: getattr(factory("x"), field)(bad)  # noqa: E731
        with self.assertRaises(TypeError) as cm:
            ctor()
        self.assertEqual(
            str(cm.exception),
            f"IntSpec.{field} must be int or None, got {type_name}",
        )


class TestIntSpecCompile(TestCase):
    """IntSpec + compile integration via the test-local
    :func:`_compile_with_dynamic_shapes` helper — graph inspection and
    precedence tests.

    The real ``torch.compile``-level ``dynamic_shapes`` surface is deferred to
    the integration PR; PR 1 keeps the behavior demonstrable here through the
    helper.
    """

    @skipIfTorchDynamo()
    def test_static_graph_has_concrete_shape(self):
        """STATIC dim appears as a concrete int in the captured graph; each
        distinct shape yields a new graph."""
        backend = EagerAndRecordGraphs()
        fn = _compile_with_dynamic_shapes(
            lambda x: x + 1,
            {"x": {0: IntSpec.static()}},
            backend=backend,
        )
        fn(torch.randn(4, 3))
        fn(torch.randn(8, 3))
        fn(torch.randn(4, 3))  # cache hit

        # STATIC bakes the concrete size into the trace: 2 distinct sizes
        # (4, 8) → 2 graphs; the repeat size=4 reuses graph #1.
        self.assertEqual(len(backend.graphs), 2)
        for gm in backend.graphs:
            shape = _tensor_placeholder_shape(gm)
            self.assertIsInstance(shape[0], int)

    @skipIfTorchDynamo()
    def test_backed_graph_has_backed_symbol(self):
        """BACKED dim appears as a backed SymInt in the final graph.

        Exactly one graph is recorded for ``[4, 8, 16, 32, 64]``: the
        scaffolding's ``maybe_mark_dynamic(tensor, 0)`` fires on call 1
        *before* dynamo traces, so dim 0 is already marked weak-dynamic
        at the first trace — dynamo promotes to a backed SymInt
        immediately, skipping the "specialize first, promote on variation"
        path that auto-dynamic would otherwise take. Calls 2-5 cache-hit
        the one compile.

        Consistent with ``test_backed_branching_bounded_recompiles`` which
        measures exactly 2 compiles (one per branch path) — confirming
        there's no initial specialization step eating a compile.
        """
        backend = EagerAndRecordGraphs()
        fn = _compile_with_dynamic_shapes(
            lambda x: x.sum(0),
            {"x": {0: IntSpec.backed("batch")}},
            backend=backend,
        )
        for n in [4, 8, 16, 32, 64]:
            fn(torch.randn(n, 3))

        # Pre-trace maybe_mark_dynamic → backed from call 1, no branches,
        # no 0/1 specialization → 1 compile covers all 5 sizes.
        self.assertEqual(len(backend.graphs), 1)
        shape = _tensor_placeholder_shape(backend.graphs[0])
        self.assertIsInstance(shape[0], torch.SymInt)
        # backed symbol: no free unbacked symbols
        self.assertEqual(len(free_unbacked_symbols(shape[0])), 0)

    @skipIfTorchDynamo()
    def test_unbacked_graph_has_unbacked_symbol(self):
        """UNBACKED dim appears as an unbacked SymInt; single compile covers all shapes."""
        backend = EagerAndRecordGraphs()
        fn = _compile_with_dynamic_shapes(
            lambda x: x.sum(0),
            {"x": {0: IntSpec.unbacked("batch")}},
            backend=backend,
        )
        for n in [4, 8, 16, 32]:
            fn(torch.randn(n, 3))

        # UNBACKED dim produces a symbol with no backing value; no guards
        # can attach (no branch on size), and 0/1 specialization doesn't
        # apply to unbacked → 1 compile covers all 4 sizes.
        self.assertEqual(len(backend.graphs), 1)
        shape = _tensor_placeholder_shape(backend.graphs[0])
        self.assertIsInstance(shape[0], torch.SymInt)
        self.assertGreater(len(free_unbacked_symbols(shape[0])), 0)

    @_fx_experimental_config.patch(no_data_dependent_graph_break=True)
    def test_unbacked_raises_dde_on_branching(self):
        """A function that branches on size(0) must raise a data-dependent
        error when that dim is marked UNBACKED.

        By default dynamo catches ``GuardOnDataDependentSymNode`` and
        re-raises as ``UserError`` (see
        ``torch/_dynamo/variables/tensor.py:2251-2259``), losing structural
        access to the original ``cond`` sympy expression. The
        ``no_data_dependent_graph_break`` config flag on
        ``torch.fx.experimental._config`` disables that rewrap, so the raw
        exception surfaces and we can assert on ``.cond`` directly —
        matching the canonical pattern used in
        ``test/test_dynamic_shapes.py``.
        """

        def fn(x):
            if x.size(0) > 5:
                return x + 1
            return x - 1

        compiled = _compile_with_dynamic_shapes(
            fn,
            {"x": {0: IntSpec.unbacked()}},
            backend="eager",
            fullgraph=True,
        )
        with self.assertRaises(GuardOnDataDependentSymNode) as cm:
            compiled(torch.randn(10, 3))
        # The guard expression must reference a free unbacked symbol —
        # that's what makes it "data-dependent" and confirms our UNBACKED
        # spec actually produced an unbacked dim (backed would be ``s0``,
        # never raise DDE).
        free_syms = cm.exception.cond.free_symbols
        self.assertEqual(len(free_syms), 1)
        # ShapeEnv names unbacked symbols with a ``u`` prefix.
        (sym,) = free_syms
        self.assertTrue(
            str(sym).startswith("u"),
            msg=f"expected unbacked symbol (u-prefix), got {sym!r}",
        )

    @skipIfTorchDynamo()
    def test_backed_branching_bounded_recompiles(self):
        """BACKED + Python branch on size: exactly 2 compiles for the
        sequence ``[4, 8, 16, 32, 64]`` against branch ``size(0) > 8``.

        The scaffolding calls ``maybe_mark_dynamic(tensor, 0)`` on every
        call before handing off to the compiled fn. On the first call that
        means dim 0 is already a backed SymInt when dynamo starts tracing
        — no initial specialization step. From there the branch splits the
        trace into two guarded compiles:

        - Call n=4:  dim 0 backed; hint=4, branch False → ``sum(1)``.
                     Guard installed: ``size <= 8``. **Compile #1**.
        - Call n=8:  size=8 satisfies ``size <= 8`` → cache hit on #1.
        - Call n=16: guard ``size <= 8`` fails → recompile with hint=16,
                     branch True → ``sum(0)``. Guard installed:
                     ``size > 8``. **Compile #2**.
        - Call n=32: size=32 satisfies ``size > 8`` → cache hit on #2.
        - Call n=64: size=64 satisfies ``size > 8`` → cache hit on #2.

        Total = 2. One trace per distinct *branch path*, regardless of the
        number of distinct shapes. Contrast with
        ``test_static_branching_recompiles_per_shape`` (5 compiles).
        """
        cnt = torch._dynamo.testing.CompileCounter()
        fn = _compile_with_dynamic_shapes(
            lambda x: x.sum(0 if x.size()[0] > 8 else 1),
            {"x": {0: IntSpec.backed("batch")}},
            backend=cnt,
        )
        for n in [4, 8, 16, 32, 64]:
            fn(torch.randn(n, 3))
        # Dim is backed from call 1; the ``> 8`` branch splits compiles by
        # path — 2 paths (True/False) → 2 compiles, regardless of shape count.
        self.assertEqual(cnt.frame_count, 2)

    @skipIfTorchDynamo()
    def test_static_branching_recompiles_per_shape(self):
        """STATIC + Python branch: exactly 5 compiles for the sequence
        ``[4, 8, 16, 32, 64]``.

        Each call sees ``size(0)`` as a concrete int baked into the trace,
        so the guard is ``size == n`` (exact match). No two calls share
        a guard, so each distinct shape forces a fresh compile — 5 shapes,
        5 compiles. The branch itself doesn't matter; even without it
        STATIC would compile 5 times.
        """
        cnt = torch._dynamo.testing.CompileCounter()
        fn = _compile_with_dynamic_shapes(
            lambda x: x.sum(0 if x.size()[0] > 8 else 1),
            {"x": {0: IntSpec.static()}},
            backend=cnt,
        )
        for n in [4, 8, 16, 32, 64]:
            fn(torch.randn(n, 3))
        # STATIC pins each concrete size into its own trace; 5 distinct
        # sizes → 5 compiles. Branch adds no compiles — STATIC would be 5
        # even without the branch.
        self.assertEqual(cnt.frame_count, 5)

    @skipIfTorchDynamo()
    def test_backed_zero_one_specialization(self):
        """BACKED symbols are specialized at 0 and 1 *unconditionally*,
        regardless of the dynamic mark — this is a PyTorch-wide rule,
        not spec-specific. For the sequence ``[0, 1, 2, 4, 8]``:

        - n=0: 0/1-spec forces concrete size. **Compile #1** (size=0).
        - n=1: 0/1-spec forces concrete size. **Compile #2** (size=1).
        - n=2: first non-{0,1} call. Scaffolding's ``maybe_mark_dynamic``
               plus dynamo's auto-dynamic (it has seen sizes 0, 1 already)
               promote the dim to a backed SymInt with guard ``size >= 2``.
               **Compile #3** (dynamic).
        - n=4: size=4 satisfies ``size >= 2`` → cache hit on #3.
        - n=8: cache hit on #3.

        Total = 3. Documents the 0/1 rule so future readers don't assume
        BACKED means "one compile for all sizes."
        """
        cnt = torch._dynamo.testing.CompileCounter()
        fn = _compile_with_dynamic_shapes(
            lambda x: x + 1,
            {"x": {0: IntSpec.backed("batch")}},
            backend=cnt,
        )
        for n in [0, 1, 2, 4, 8]:
            fn(torch.randn(n, 3))
        # 0 and 1 each force their own specialized compile (PyTorch-wide
        # 0/1 rule, applies even to backed); sizes ≥ 2 share one backed
        # compile with guard ``size >= 2`` → 2 + 1 = 3 compiles.
        self.assertEqual(cnt.frame_count, 3)

    @skipIfTorchDynamo()
    def test_backed_equality_branching(self):
        """BACKED + Python branch on ``==``: the guard is a *point*
        specialization, not an inequality range, producing different
        cache behavior from the ``>`` case.

        For ``x.size(0) == 3`` against sizes ``[3, 4, 5, 6]``:

        - n=3: branch True. Guard installed: ``size == 3`` (point).
               **Compile #1** — effectively specialized to 3.
        - n=4: guard ``size == 3`` fails → recompile. Branch False.
               Guard ``size != 3``, dim remains backed for 4-and-above.
               **Compile #2** (dynamic, else-branch).
        - n=5: satisfies ``size != 3`` → cache hit on #2.
        - n=6: cache hit on #2.

        Total = 2. Same count as the ``>`` case, but the compiles carry
        different guards: #1 here is a single-size specialization (not a
        range like ``size <= 8``), which means repeating n=3 later reuses
        #1 while any other size reuses #2.
        """
        cnt = torch._dynamo.testing.CompileCounter()
        fn = _compile_with_dynamic_shapes(
            lambda x: x + 1 if x.size()[0] == 3 else x - 1,
            {"x": {0: IntSpec.backed("batch")}},
            backend=cnt,
        )
        for n in [3, 4, 5, 6]:
            fn(torch.randn(n, 3))
        # ``== 3`` splits into two branch paths: point-specialization at
        # size=3 (compile #1) and the ``size != 3`` backed compile
        # (compile #2). Same count as the ``>`` case, different guards.
        self.assertEqual(cnt.frame_count, 2)

    @skipIfTorchDynamo()
    def test_static_precedence_over_dynamic_true(self):
        """IntSpec.static() must win over compile(dynamic=True)."""
        cnt = torch._dynamo.testing.CompileCounter()
        fn = _compile_with_dynamic_shapes(
            lambda x: x + 1,
            {"x": {0: IntSpec.static()}},
            backend=cnt,
            dynamic=True,
        )
        fn(torch.randn(4, 3))
        fn(torch.randn(8, 3))
        # Spec STATIC beats ``compile(dynamic=True)``: 2 distinct shapes
        # each get a specialized compile → 2 compiles. If ``dynamic=True``
        # had won, we'd see 1 compile with a backed SymInt.
        self.assertEqual(cnt.frame_count, 2)

    @skipIfTorchDynamo()
    def test_backed_precedence_over_dynamic_false(self):
        """IntSpec.backed() must win over compile(dynamic=False).

        With the compile-context integration, the spec selects
        DimDynamic.DYNAMIC directly — the first call is already backed, no
        initial specialization, so a single compile covers all shapes.
        """
        cnt = torch._dynamo.testing.CompileCounter()
        fn = _compile_with_dynamic_shapes(
            lambda x: x.sum(0),
            {"x": {0: IntSpec.backed("batch")}},
            backend=cnt,
            dynamic=False,
        )
        for n in [4, 8, 16, 32, 64]:
            fn(torch.randn(n, 3))
        # Spec BACKED beats ``compile(dynamic=False)``: dim is backed from
        # call 1 via the pre-trace mark, no branches, no 0/1 sizes → 1
        # compile. If ``dynamic=False`` had won, we'd see 5 (one per shape).
        self.assertEqual(cnt.frame_count, 1)

    @skipIfTorchDynamo()
    def test_list_form(self):
        """List-form per-dim spec: dim 0 BACKED, dim 1 STATIC.

        Two assertions, one per dim:

        1. **Dim 0 (BACKED) absorbs shape changes** — varying dim 0 while
           keeping dim 1 fixed reuses the same compile; graph count stays
           at 1.
        2. **Dim 1 (STATIC) forces recompiles** — varying dim 1 while
           keeping dim 0 fixed bumps the graph count by one per distinct
           dim-1 value.

        Also inspects the final captured graph to confirm the kinds:
        dim 0 is a backed ``SymInt``, dim 1 is a concrete ``int``.
        """
        backend = EagerAndRecordGraphs()
        fn = _compile_with_dynamic_shapes(
            lambda x: x + 1,
            {"x": [IntSpec.backed("batch"), IntSpec.static()]},
            backend=backend,
        )

        # Call 1 — initial compile.
        fn(torch.randn(4, 3))
        self.assertEqual(len(backend.graphs), 1)

        # Vary dim 0 only (BACKED absorbs it → no new compile).
        fn(torch.randn(8, 3))
        fn(torch.randn(16, 3))
        self.assertEqual(len(backend.graphs), 1)

        # Vary dim 1 (STATIC pins it → each distinct value recompiles).
        fn(torch.randn(16, 5))
        self.assertEqual(len(backend.graphs), 2)
        fn(torch.randn(16, 7))
        self.assertEqual(len(backend.graphs), 3)

        # The last captured graph: dim 0 backed SymInt, dim 1 concrete int=7.
        shape = _tensor_placeholder_shape(backend.graphs[-1])
        self.assertIsInstance(shape[0], torch.SymInt)
        self.assertEqual(len(free_unbacked_symbols(shape[0])), 0)
        self.assertIsInstance(shape[1], int)
        self.assertEqual(shape[1], 7)

    def test_none_entry_inherits_context(self):
        """A None entry in a list-form spec should not mark the dim."""
        fn = _compile_with_dynamic_shapes(
            lambda x: x + 1,
            {"x": [IntSpec.backed("batch"), None]},
            backend="eager",
        )
        x = torch.randn(4, 3)
        self.assertEqual(fn(x), x + 1)


if __name__ == "__main__":
    run_tests()
