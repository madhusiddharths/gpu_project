import pytest
import warp as wp

from warp_dem import DeviceError, assert_same_device, describe_device, resolve_device


def test_cpu_always_available():
    d = resolve_device("cpu")
    assert not d.is_cuda


def test_auto_returns_a_device():
    d = resolve_device("auto")
    assert d is not None


def test_auto_prefers_cuda_when_present():
    d = resolve_device("auto")
    if wp.get_cuda_devices():
        assert d.is_cuda
    else:
        assert not d.is_cuda


def test_bad_name_raises():
    with pytest.raises(DeviceError):
        resolve_device("gpu_please")


def test_explicit_cuda_raises_when_absent():
    """On a CPU-only machine, asking for CUDA must fail loudly, not fall back."""
    if wp.get_cuda_devices():
        pytest.skip("CUDA is available on this machine")
    with pytest.raises(DeviceError):
        resolve_device("cuda:0")


def test_describe_device_returns_string():
    assert isinstance(describe_device(resolve_device("cpu")), str)


def test_assert_same_device_accepts_matching():
    d = resolve_device("cpu")
    a = wp.zeros(4, dtype=float, device=d)
    b = wp.zeros(4, dtype=float, device=d)
    assert str(assert_same_device(a, b)) == str(d)