# NIXL Bug: Raw address registration produces broken rkeys for RDMA

## Summary

Registering memory with raw address descriptors (Python tuples or numpy arrays) through `register_memory()` produces rkeys that fail RDMA reads with `NIXL_ERR_REMOTE_DISCONNECT`. The same memory registered via torch tensor objects works correctly.

## Environment

- NIXL 0.8.0 (from `nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.8.0`)
- nScale B200 cluster, ConnectX-8 InfiniBand, UCX transport (`rc_x,rc,dc_x,dc,cuda_copy`)
- Cross-node RDMA over InfiniBand (not loopback)

## Reproduction

Two NIXL agents on different nodes. Source registers GPU memory and starts listen thread. Target fetches metadata via `fetch_remote_metadata`, then attempts RDMA READ.

### Works: tensor registration

```python
# Source side
tensors = [tensor_a, tensor_b, tensor_c]  # torch.Tensor objects on CUDA
agent.register_memory(tensors, backends=["UCX"])
```

Result: target RDMA READ succeeds, 199 Gbps throughput.

### Fails: raw tuple registration (same memory)

```python
# Source side - same GPU addresses, registered as raw tuples
descs = [(tensor_a.data_ptr(), tensor_a.numel() * tensor_a.element_size(), 0, "cuda"),
         (tensor_b.data_ptr(), tensor_b.numel() * tensor_b.element_size(), 0, "cuda")]
agent.register_memory(descs, mem_type="cuda", backends=["UCX"])
```

Result: target RDMA READ fails with `NIXL_ERR_REMOTE_DISCONNECT`.

### Also fails: numpy array registration (same memory)

```python
import numpy as np
descs = np.array([
    [tensor_a.data_ptr(), tensor_a.numel() * tensor_a.element_size(), 0],
    [tensor_b.data_ptr(), tensor_b.numel() * tensor_b.element_size(), 0],
], dtype=np.uint64)
agent.register_memory(descs, mem_type="cuda", backends=["UCX"])
```

Result: same `NIXL_ERR_REMOTE_DISCONNECT`.

## Analysis

Looking at `nixl_agent.get_reg_descs()`, the tensor path and tuple/numpy paths construct `nixlRegDList` differently:

**Tensor path** (works):
- Iterates tensors, extracts `(data_ptr(), numel() * element_size(), get_device())` into an Nx3 uint64 numpy array
- Infers `mem_type` from device ("cuda" or "cpu")
- Calls `nixlRegDList(mem_type, numpy_array)`

**Tuple path** (broken):
- Passes 4-element tuples `(addr, size, device_id, "")` directly
- Calls `nixlRegDList(mem_type, tuples)`

**Numpy path** (broken):
- Passes Nx3 uint64 numpy array (same format as tensor path)
- Calls `nixlRegDList(mem_type, numpy_array)` - identical call to tensor path

The numpy path is especially surprising since it produces the same `nixlRegDList` constructor call as the tensor path. The difference might be in how the C++ binding handles the mem_type parameter vs inferring it from tensors, or in some CUDA context state that the tensor path implicitly ensures.

## Use case

We want to register contiguous memory pools (multiple tensors allocated adjacently) as single NIXL registrations to reduce registration overhead. Moein recommended this approach: "register the full buffer pool, not per buffer." Raw address registration is needed because we're computing pool boundaries from individual tensor addresses - there's no single torch tensor covering the pool.

## Workaround

Create a `torch.uint8` tensor view spanning each pool using `as_strided` on the first tensor in the region, then register those views through the tensor code path. This is functionally correct but fragile (relies on `as_strided` not bounds-checking).

## Expected behavior

All three registration methods (tensor, tuple, numpy) should produce identical rkeys for the same GPU memory addresses. RDMA reads should work regardless of which path was used to register.
