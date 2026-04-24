# Owner(s): ["module: dtensor"]

import unittest

import torch
import torch.distributed as dist
from torch.distributed._local_tensor import LocalTensorMode
from torch.distributed.spmd_types import is_available as spmd_types_available

if spmd_types_available():
    from torch.distributed.spmd_types import (
        _reset,
        all_gather,
        all_reduce,
        assert_type,
        convert,
        get_axis_local_type,
        get_partition_spec,
        I,
        no_typecheck,
        normalize_axis,
        P,
        PartitionSpec,
        R,
        reduce_scatter,
        reinterpret,
        S,
        SpmdTypeError,
        typecheck,
        V,
    )

from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed.fake_pg import FakeStore


# ---------------------------------------------------------------------------
# Type system (no distributed setup)
# ---------------------------------------------------------------------------


@unittest.skipUnless(spmd_types_available(), "requires spmd_types")
class TestTypeSystem(TestCase):
    def test_enum_identity_and_aliases(self):
        from torch.distributed.spmd_types import Invariant, Partial, Replicate, Varying  # noqa: F811

        self.assertIs(Replicate, R)
        self.assertIs(Invariant, I)
        self.assertIs(Varying, V)
        self.assertIs(Partial, P)

    def test_shard_equality(self):
        self.assertEqual(S(0), S(0))
        self.assertNotEqual(S(0), S(1))
        self.assertNotEqual(S(0), V)

    def test_backward_types(self):
        self.assertIs(R.backward_type(), P)
        self.assertIs(P.backward_type(), R)
        self.assertIs(I.backward_type(), I)
        self.assertIs(V.backward_type(), V)
        self.assertEqual(S(0).backward_type(), S(0))


# ---------------------------------------------------------------------------
# Distributed tests (fake PG + LocalTensorMode)
# ---------------------------------------------------------------------------

WORLD_SIZE = 2


def _setup_fake_dist(world_size=WORLD_SIZE):
    if dist.is_initialized():
        dist.destroy_process_group()
    _reset()
    store = FakeStore()
    dist.init_process_group(backend="fake", rank=0, world_size=world_size, store=store)
    return dist.distributed_c10d._get_default_group()


def _teardown_fake_dist():
    if dist.is_initialized():
        dist.destroy_process_group()
    _reset()


@unittest.skipUnless(spmd_types_available(), "requires spmd_types")
class TestSpmdTypes(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pg = _setup_fake_dist()

    @classmethod
    def tearDownClass(cls):
        _teardown_fake_dist()

    def test_local_spmd_types(self):
        # V @ R -> V
        with LocalTensorMode(WORLD_SIZE) as mode, typecheck():
            x = mode.rank_map(lambda r: torch.randn(2, 4))
            assert_type(x, {self.pg: R})
            y = mode.rank_map(lambda r: torch.randn(4, 3))
            assert_type(y, {self.pg: V})
            result = torch.matmul(x, y)
        self.assertIs(get_axis_local_type(result, self.pg), V)

        # P @ P is rejected
        with LocalTensorMode(WORLD_SIZE) as mode, typecheck():
            x = mode.rank_map(lambda r: torch.randn(2, 4))
            assert_type(x, {self.pg: P})
            y = mode.rank_map(lambda r: torch.randn(4, 3))
            assert_type(y, {self.pg: P})
            with self.assertRaises(SpmdTypeError):
                torch.matmul(x, y)

    def test_reinterpret(self):
        # reinterpret(R -> V)
        with LocalTensorMode(WORLD_SIZE) as mode:
            base = torch.randn(4)
            x = mode.rank_map(lambda r: base.clone())
            assert_type(x, {self.pg: R})
            with typecheck():
                result = reinterpret(x, self.pg, src=R, dst=V, expert_mode=True)
            self.assertIs(get_axis_local_type(result, self.pg), V)
            for r in range(WORLD_SIZE):
                torch.testing.assert_close(result._local_tensors[r], base)

        # reinterpret(R -> S(0)) rejected
        with LocalTensorMode(WORLD_SIZE) as mode:
            x = mode.rank_map(lambda r: torch.randn(4))
            assert_type(x, {self.pg: R})
            with self.assertRaises(ValueError, msg="does not support S(i)"):
                reinterpret(x, self.pg, src=R, dst=S(0))

    def test_convert(self):
        # convert(R -> V)
        with LocalTensorMode(WORLD_SIZE) as mode:
            base = torch.arange(4, dtype=torch.float).reshape(WORLD_SIZE, 2)
            x = mode.rank_map(lambda r: base.clone())
            assert_type(x, {self.pg: R})
            with typecheck():
                result = convert(x, self.pg, src=R, dst=V)
            self.assertIs(get_axis_local_type(result, self.pg), V)
            for r in range(WORLD_SIZE):
                torch.testing.assert_close(result._local_tensors[r], base[r])

        # convert(R -> S(0))
        with LocalTensorMode(WORLD_SIZE) as mode:
            base = torch.arange(4, dtype=torch.float)
            x = mode.rank_map(lambda r: base.clone())
            assert_type(x, {self.pg: R})
            with typecheck():
                result = convert(x, self.pg, src=R, dst=S(0))
            for r in range(WORLD_SIZE):
                expected = base[r * 2 : (r + 1) * 2]
                torch.testing.assert_close(result._local_tensors[r], expected)

    def test_collectives(self):
        # all_reduce(P -> R)
        with LocalTensorMode(WORLD_SIZE) as mode:
            x = mode.rank_map(lambda r: torch.tensor([float(r)]))
            assert_type(x, {self.pg: P})
            with typecheck():
                result = all_reduce(x, self.pg, src=P, dst=R)
            self.assertIs(get_axis_local_type(result, self.pg), R)
            expected = torch.tensor([float(sum(range(WORLD_SIZE)))])
            for r in range(WORLD_SIZE):
                torch.testing.assert_close(result._local_tensors[r], expected)

        # all_gather(V -> R)
        with LocalTensorMode(WORLD_SIZE) as mode:
            x = mode.rank_map(lambda r: torch.tensor(float(r)))
            assert_type(x, {self.pg: V})
            with typecheck():
                result = all_gather(x, self.pg, src=V, dst=R)
            self.assertIs(get_axis_local_type(result, self.pg), R)
            expected = torch.arange(WORLD_SIZE, dtype=torch.float)
            for r in range(WORLD_SIZE):
                torch.testing.assert_close(result._local_tensors[r], expected)

        # reduce_scatter(P -> S(0))
        with LocalTensorMode(WORLD_SIZE) as mode:
            x = mode.rank_map(
                lambda r: torch.arange(WORLD_SIZE * 2, dtype=torch.float) + r
            )
            assert_type(x, {self.pg: P})
            with typecheck():
                result = reduce_scatter(x, self.pg, src=P, dst=S(0))
            for r in range(WORLD_SIZE):
                chunk_size = 2
                expected = sum(
                    (torch.arange(WORLD_SIZE * 2, dtype=torch.float) + src)[
                        r * chunk_size : (r + 1) * chunk_size
                    ]
                    for src in range(WORLD_SIZE)
                )
                torch.testing.assert_close(result._local_tensors[r], expected)

    def test_global_spmd(self):
        tp = normalize_axis(self.pg)

        with LocalTensorMode(WORLD_SIZE) as mode, typecheck(local=False):

            def rank_map(cb):
                with no_typecheck():
                    return mode.rank_map(cb)

            def make_input(shape, typ):
                if typ is R or typ is I:
                    base = torch.randn(shape)
                    result = rank_map(lambda r: base.clone())
                else:
                    result = rank_map(lambda r: torch.randn(shape) + r)
                assert_type(result, {tp: typ})
                return result

            # unary S(0) propagation
            x = make_input((4, 3), S(0))
            result = torch.neg(x)
            self.assertEqual(get_partition_spec(result), PartitionSpec(tp, None))

            # binary S(0)+S(0)
            y = make_input((4, 3), S(0))
            result = torch.add(x, y)
            self.assertEqual(get_partition_spec(result), PartitionSpec(tp, None))

            # S(0)+R rejected
            z = make_input((4, 3), R)
            with self.assertRaises(SpmdTypeError):
                x + z

            # mm(S(0), R) -> S(0)
            x = make_input((4, 3), S(0))
            w = make_input((3, 5), R)
            result = torch.mm(x, w)
            self.assertEqual(get_partition_spec(result), PartitionSpec(tp, None))

            # mm(S(1), S(0)) -> Partial
            x = make_input((4, 3), S(1))
            w = make_input((3, 5), S(0))
            result = torch.mm(x, w)
            self.assertIs(get_axis_local_type(result, tp), P)


if __name__ == "__main__":
    run_tests()
