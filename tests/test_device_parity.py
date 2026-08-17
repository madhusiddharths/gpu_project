"""CPU/CUDA parity — the test that guards the project's central claim.

Skips entirely on CPU-only machines. Run these at every phase boundary on a
GPU box, not just at the end.
"""

import numpy as np
import pytest
import warp as wp

from warp_dem import resolve_device

pytestmark = pytest.mark.skipif(
    not wp.get_cuda_devices(), reason="no CUDA device on this machine"
)

# Tolerance, not equality. Float addition is not associative, and atomic
# accumulation order on the GPU is not deterministic — so identical results
# across devices are neither expected nor required.
RTOL = 1e-5
ATOL = 1e-6


@wp.kernel
def _axpy(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), k: float,
          out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    out[i] = a[i] * k + b[i]


@wp.kernel
def _scatter_add(idx: wp.array(dtype=int), vals: wp.array(dtype=float),
                 out: wp.array(dtype=float)):
    """Many threads accumulate into few slots — the force-accumulation pattern."""
    i = wp.tid()
    wp.atomic_add(out, idx[i], vals[i])


def _run(kernel, dim, inputs_fn, out_shape, out_dtype, device):
    d = resolve_device(device)
    inputs, out = inputs_fn(d, out_shape, out_dtype)
    wp.launch(kernel=kernel, dim=dim, inputs=inputs + [out], device=d)
    wp.synchronize_device(d)
    return out.numpy()


def test_elementwise_parity():
    n = 4096
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((n, 3)).astype(np.float32)
    b_np = rng.standard_normal((n, 3)).astype(np.float32)

    def make(d, shape, dtype):
        return (
            [wp.array(a_np, dtype=wp.vec3, device=d),
             wp.array(b_np, dtype=wp.vec3, device=d),
             2.5],
            wp.zeros(n, dtype=wp.vec3, device=d),
        )

    cpu = _run(_axpy, n, make, n, wp.vec3, "cpu")
    gpu = _run(_axpy, n, make, n, wp.vec3, "cuda:0")
    np.testing.assert_allclose(cpu, gpu, rtol=RTOL, atol=ATOL)


def test_atomic_accumulation_parity():
    """The pattern that breaks on GPU if written naively."""
    n, slots = 100_000, 64
    rng = np.random.default_rng(1)
    idx_np = rng.integers(0, slots, size=n).astype(np.int32)
    val_np = rng.standard_normal(n).astype(np.float32)

    def make(d, shape, dtype):
        return (
            [wp.array(idx_np, dtype=int, device=d),
             wp.array(val_np, dtype=float, device=d)],
            wp.zeros(slots, dtype=float, device=d),
        )

    cpu = _run(_scatter_add, n, make, slots, float, "cpu")
    gpu = _run(_scatter_add, n, make, slots, float, "cuda:0")

    # Looser tolerance: 100k additions into 64 slots in nondeterministic order
    # accumulates real rounding divergence. That is physics of floats, not a bug.
    np.testing.assert_allclose(cpu, gpu, rtol=1e-4, atol=1e-4)


def test_gpu_run_to_run_stability():
    """Two identical GPU runs should agree closely — but need not be identical."""
    n, slots = 100_000, 64
    rng = np.random.default_rng(2)
    idx_np = rng.integers(0, slots, size=n).astype(np.int32)
    val_np = rng.standard_normal(n).astype(np.float32)

    def make(d, shape, dtype):
        return (
            [wp.array(idx_np, dtype=int, device=d),
             wp.array(val_np, dtype=float, device=d)],
            wp.zeros(slots, dtype=float, device=d),
        )

    first = _run(_scatter_add, n, make, slots, float, "cuda:0")
    second = _run(_scatter_add, n, make, slots, float, "cuda:0")
    np.testing.assert_allclose(first, second, rtol=1e-4, atol=1e-4)


def test_auto_resolves_to_cuda_when_present():
    assert resolve_device("auto").is_cuda


def test_explicit_cpu_still_works_on_gpu_machine():
    """Forcing CPU on a GPU box is required for the Phase 8 benchmark."""
    assert not resolve_device("cpu").is_cuda