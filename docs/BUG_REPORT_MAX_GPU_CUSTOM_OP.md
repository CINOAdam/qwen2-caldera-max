# MAX GPU Custom Op Segfault (Device-Pointer Path)

## Summary

Using `max.torch.CustomOpLibrary` to run a trivial Mojo custom op on CUDA
segfaults inside `max/engine/api.py:_Model_execute`. The same custom op
works on CPU, and `max.torch.graph_op` works on CUDA. The crash happens even
for a minimal `output[0,0] = input[0,0]` kernel.

## Environment

- OS: Pop!_OS 22.04 (kernel 6.17.4-76061704-generic)
- GPU: NVIDIA RTX 3090
- Driver: 580.82.09 (nvidia-smi reports CUDA 13.0)
- Python: 3.12.9
- Torch: 2.6.0+cu124 (torch.version.cuda=12.4, cuDNN=90100)
- MAX: 26.1.0.dev2026010305
- max-core: 26.1.0.dev2026010305
- mojo: 0.26.1.0.dev2026010305

## Repro

1. Ensure CUDA is visible and the float8 alias is present (script handles it).
2. Run the repro:

```
.venv/bin/python scripts/max_gpu_custom_op_repro.py --device cuda --dtype float32 --size 2
```

Expected: prints `ok` and copied value.
Actual: process segfaults (no Python exception).

CPU path works:

```
.venv/bin/python scripts/max_gpu_custom_op_repro.py --device cpu --dtype float32 --size 2
```

Graph-op path works on CUDA (see inline test in notes below).

## Logs / Artifacts

- `artifacts/max_gpu_custom_op_gpu.log` (Python faulthandler traceback, segfault)
- `artifacts/max_gpu_custom_op_gdb.txt` (gdb backtrace, segfault in worker thread)
- `scripts/run_max_gpu_custom_op_repro.sh` captures the debug env run

## Minimal Mojo op (from repro)

```
import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor

@compiler.register("copy_first")
struct CopyFirst:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype = DType.float32, rank=2],
        input: InputTensor[dtype = DType.float32, rank=2],
        ctx: DeviceContextPtr,
    ) raises:
        output[0, 0] = input[0, 0]
```

## Notes

- `max.driver.Tensor.from_dlpack` works on CUDA for basic tensors.
- `max.torch.graph_op` works on CUDA for a trivial op:

```
import torch, numpy as np, max.torch
from max.dtype import DType
from max.graph import ops

if not hasattr(torch, "float8_e8m0fnu") and hasattr(torch, "float8_e5m2"):
    torch.float8_e8m0fnu = torch.float8_e5m2

@max.torch.graph_op
def max_copy(x):
    return x + ops.constant(np.array(0, dtype=np.int32), dtype=DType.int32, device=x.device)

x = torch.arange(9, device="cuda", dtype=torch.int32).reshape(3, 3)
out = torch.empty_like(x)
    max_copy(out, x)
```

- Using `graph_op(kernel_library=...)` with `ops.custom("copy_first", ...)` also
  segfaults on CUDA (same stack in `max/engine/api.py`), suggesting the issue
  is tied to custom kernels compiled via `@compiler.register`, not DLPack.
