import warp as wp
import numpy as np

wp.init()
print("Warp version:", wp.config.version)
print("Devices:", wp.get_devices())
print("CUDA devices:", wp.get_cuda_devices())

@wp.kernel
def scale(a: wp.array(dtype=float), out: wp.array(dtype=float), k: float):
    i = wp.tid()              # this thread's index
    out[i] = a[i] * k

device = "cpu"
a = wp.array(np.arange(8, dtype=np.float32), dtype=float, device=device)
out = wp.zeros(8, dtype=float, device=device)

wp.launch(kernel=scale, dim=8, inputs=[a, out, 3.0], device=device)
print("Result:", out.numpy())