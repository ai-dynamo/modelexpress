# PR 19920: [1/2] Add ModelExpress coordination for remote instance weight loading - matching TP

## Executive Summary

This document provides a design review and feedback for SGLang PR 19920, which adds ModelExpress coordination for remote instance weight loading. The PR integrates ModelExpress gRPC server as a coordination layer for TransferEngine-based weight transfers, replacing direct HTTP communication between seed and target instances.

**Key Changes:**
- Adds `MODEL_EXPRESS` backend option for `remote_instance_weight_loader_backend`
- Integrates ModelExpress client for metadata coordination
- Supports TP rank matching between seed and target instances
- Uses TransferEngine for actual RDMA transfers (coordinated via ModelExpress)

## Architecture Overview

### Current Flow (Before PR 19920)

**NCCL Backend:**
```
Seed Instance → Direct HTTP → Target Instance
  - Seed publishes TransferEngine session ID via HTTP endpoint
  - Target queries seed HTTP endpoint for session ID
  - Target connects directly to seed via TransferEngine
```

**TransferEngine Backend (Direct):**
```
Seed Instance → HTTP endpoint → Target Instance
  - Seed exposes /get_remote_instance_transfer_engine_info
  - Target queries per-rank session IDs
  - Direct TransferEngine connection
```

### New Flow (After PR 19920)

**ModelExpress Backend:**
```
Seed Instance → ModelExpress Server → Target Instance
  - Seed publishes metadata to ModelExpress gRPC server
  - Target queries ModelExpress for seed metadata
  - ModelExpress coordinates ready state
  - Target connects to seed via TransferEngine (using session ID from metadata)
```

## Implementation Review

### 1. Protocol Buffer Integration

#### ✅ **Good: Correct Use of `oneof` Pattern**

**File**: `python/sglang/srt/model_loader/loader.py` (line ~2340)

The implementation correctly uses the `oneof` pattern to extract TransferEngine session ID:

```python
backend_field = source_worker.WhichOneof("backend_metadata")
if backend_field == "transfer_engine_session_id":
    seed_session_id = source_worker.transfer_engine_session_id
else:
    raise RuntimeError(
        f"ModelExpress: expected transfer_engine_session_id, "
        f"got backend_metadata={backend_field}"
    )
```

**Comment**: This correctly handles the `oneof` pattern from ModelExpress PR 157. Good error handling when the wrong backend type is present.

#### ⚠️ **Concern: No Fallback for NIXL Backend**

**Issue**: The code only handles `transfer_engine_session_id` and raises an error for other backends. What if the source uses NIXL backend?

**Recommendation**: Add support for NIXL backend or provide a clear error message:

```python
backend_field = source_worker.WhichOneof("backend_metadata")
if backend_field == "transfer_engine_session_id":
    seed_session_id = source_worker.transfer_engine_session_id
elif backend_field == "nixl_metadata":
    raise RuntimeError(
        f"ModelExpress: source worker {tp_rank} uses NIXL backend, "
        f"but MODEL_EXPRESS backend requires TransferEngine. "
        f"Please use a source with TransferEngine backend or use NIXL directly."
    )
else:
    raise RuntimeError(
        f"ModelExpress: unknown backend_metadata={backend_field} "
        f"for worker {tp_rank}"
    )
```

### 2. Source Side: Publishing Metadata

#### ✅ **Good: Proper Metadata Publishing**

**File**: `python/sglang/srt/model_executor/model_runner.py` (line ~680-750)

The `_publish_model_express_metadata()` function:
- Correctly builds tensor descriptors from weight info
- Uses `transfer_engine_session_id` in the `oneof` field
- Publishes both metadata and ready flag
- Handles element size to dtype mapping for FP8 models

**Comment**: The implementation correctly uses byte sizes (`numel * element_size`) for tensor descriptors, which is important for mixed-dtype models (FP8 + BF16).

#### ⚠️ **Concern: Dtype Inference from Element Size**

**File**: `python/sglang/srt/model_executor/model_runner.py` (line ~700)

```python
element_size_to_dtype = {1: "float8_e4m3fn", 2: "bfloat16", 4: "float32", 8: "float64"}
```

**Issue**: This mapping is lossy. Multiple dtypes can have the same element size:
- Element size 2: `float16`, `bfloat16`, `int16`, `uint16`
- Element size 1: `int8`, `uint8`, `float8_e4m3fn`, `float8_e5m2`

**Recommendation**: Use actual tensor dtype instead of inferring from element size:

```python
tensors = []
for name, (addr, numel, element_size) in weight_info.items():
    # Get actual tensor to determine dtype
    tensor = dict(model.named_parameters())[name]
    dtype_str = str(tensor.dtype).replace("torch.", "")
    
    tensors.append(p2p_pb2.TensorDescriptor(
        name=name,
        addr=addr,
        size=numel * element_size,
        device_id=self.gpu_id,
        dtype=dtype_str,  # Use actual dtype
    ))
```

**Alternative**: If weight_info doesn't include tensor references, add dtype to the weight_info tuple:
```python
# In register_memory_region, return (addr, numel, element_size, dtype_str)
weight_info[name] = (addr, numel, element_size, str(tensor.dtype).replace("torch.", ""))
```

#### ✅ **Good: TP Rank Matching**

**File**: `python/sglang/srt/model_loader/loader.py` (line ~2310)

The code correctly matches TP ranks:
```python
for w in response.workers:
    if w.worker_rank == tp_rank:
        source_worker = w
        break
```

This ensures each target TP rank connects to the corresponding seed TP rank, which is critical for tensor parallelism.

### 3. Target Side: Loading Weights

#### ✅ **Good: Byte Size Matching**

**File**: `python/sglang/srt/model_loader/loader.py` (line ~2370)

The code correctly uses byte sizes for matching:
```python
seed_ptr, seed_size = weight_info
local_size = tensor.numel() * tensor.element_size()
if seed_size != local_size:
    raise RuntimeError(...)
```

**Comment**: This is correct! RDMA is a memcpy operation, so byte size matching is sufficient. Dtype differences (e.g., FP8 vs BF16) are handled by the model's quantization logic, not the transfer layer.

#### ⚠️ **Concern: Missing Tensor Name Validation**

**Issue**: The code assumes tensor names match exactly between seed and target. What if:
- Model architectures differ slightly?
- Tensor names have different prefixes?
- Some tensors are missing?

**Recommendation**: Add more robust matching:

```python
for name, tensor in model.named_parameters():
    weight_info = seed_weight_info.get(name, None)
    if weight_info is None:
        # Try fuzzy matching or provide helpful error
        logger.warning(
            f"ModelExpress: tensor '{name}' not found in seed metadata. "
            f"Available tensors: {list(seed_weight_info.keys())[:10]}..."
        )
        raise RuntimeError(
            f"ModelExpress: cannot find weight info for {name} "
            f"in seed metadata. This may indicate a model architecture mismatch."
        )
```

#### ✅ **Good: Ready State Coordination**

**File**: `python/sglang/srt/model_loader/loader.py` (line ~2280)

The code correctly waits for seed ready state:
```python
ready, session_id, metadata_hash = mx_client.wait_for_ready(
    model_name, worker_id=tp_rank,
)
```

This ensures the target doesn't start transferring before the seed is fully initialized and stable.

### 4. Configuration & CLI Arguments

#### ✅ **Good: Clear CLI Arguments**

**File**: `python/sglang/srt/server_args.py`

The PR adds three new CLI arguments:
- `--model-express-url`: ModelExpress server URL
- `--model-express-model-name`: Model name for coordination
- `--model-express-source`: Flag to run as seed source

**Comment**: The arguments are well-named and follow SGLang's existing patterns.

#### ⚠️ **Concern: Validation Logic**

**File**: `python/sglang/srt/server_args.py` (line ~2722)

```python
if self.remote_instance_weight_loader_backend == "model_express":
    if self.model_express_url is None:
        logger.warning("Fallback load_format to 'auto'...")
        self.load_format = "auto"
```

**Issue**: The validation silently falls back to `auto` instead of raising an error. This could lead to confusion.

**Recommendation**: Make validation stricter or provide clearer messaging:

```python
if self.remote_instance_weight_loader_backend == "model_express":
    if self.model_express_url is None:
        raise ValueError(
            "--model-express-url is required when using "
            "--remote-instance-weight-loader-backend=model_express"
        )
    if not self.validate_transfer_engine():
        raise ValueError(
            "TransferEngine is required for model_express backend. "
            "Please install mooncake.engine or use a different backend."
        )
```

#### ⚠️ **Concern: Model Name Default**

**File**: `python/sglang/srt/model_executor/model_runner.py` (line ~685)

```python
model_name = (
    self.server_args.model_express_model_name
    or self.server_args.model_path
)
```

**Issue**: Using `model_path` as default could lead to inconsistent model names (e.g., `/path/to/model` vs `meta-llama/Llama-3.1-70B`).

**Recommendation**: Use a more consistent default or require explicit model name:

```python
model_name = self.server_args.model_express_model_name
if not model_name:
    # Extract model name from model_path (e.g., last component)
    model_name = os.path.basename(self.server_args.model_path.rstrip('/'))
    logger.warning(
        f"ModelExpress: using model_name='{model_name}' from model_path. "
        f"Consider setting --model-express-model-name explicitly."
    )
```

### 5. Error Handling

#### ✅ **Good: Comprehensive Error Messages**

The code provides clear error messages for common failure modes:
- Missing metadata
- Worker rank mismatch
- Size mismatches
- TransferEngine failures

#### ⚠️ **Concern: Timeout Handling**

**File**: `python/sglang/srt/model_loader/loader.py` (line ~2280)

```python
ready, session_id, metadata_hash = mx_client.wait_for_ready(
    model_name, worker_id=tp_rank,
)
if not ready:
    raise RuntimeError("ModelExpress: timed out waiting for seed ready...")
```

**Issue**: The timeout is not configurable and may not be visible in the error message.

**Recommendation**: Add timeout parameter and include it in error:

```python
timeout_seconds = load_config.model_express_ready_timeout or 7200  # 2 hours default
ready, session_id, metadata_hash = mx_client.wait_for_ready(
    model_name, worker_id=tp_rank, timeout_seconds=timeout_seconds,
)
if not ready:
    raise RuntimeError(
        f"ModelExpress: timed out waiting for seed ready "
        f"(model={model_name}, worker={tp_rank}, timeout={timeout_seconds}s)"
    )
```

### 6. Integration with TransferEngine

#### ✅ **Good: Reuses Existing TransferEngine Infrastructure**

The PR correctly reuses:
- `register_memory_region()` for memory registration
- `batch_transfer_sync_read()` for RDMA transfers
- Existing TransferEngine initialization logic

**Comment**: This is a clean integration that doesn't duplicate code.

#### ⚠️ **Concern: TransferEngine Initialization Timing**

**File**: `python/sglang/srt/model_executor/model_runner.py` (line ~1075)

For seed sources, TransferEngine weight info is registered in `model_specific_adjustment()`:

```python
if self.server_args.model_express_source:
    if self.remote_instance_transfer_engine_weight_info is None:
        self.remote_instance_transfer_engine_weight_info = (
            register_memory_region(self.model, self.remote_instance_transfer_engine)
        )
    self._publish_model_express_metadata()
```

**Issue**: This happens after model loading. If the model is loaded via `DefaultModelLoader` (load_format=auto), the weights may have been processed/quantized, which could affect memory addresses.

**Recommendation**: Document this timing and ensure weights are stable before registration:

```python
# Ensure model weights are finalized before registering
# (post_load_weights may modify weights)
if hasattr(self.model, "post_load_weights"):
    self.model.post_load_weights()

# Now register memory regions (weights are stable)
if self.server_args.model_express_source:
    ...
```

### 7. Testing & Edge Cases

#### ❓ **Missing: Test Coverage**

**Questions**:
1. Are there unit tests for `load_model_from_model_express()`?
2. Are there integration tests for the full flow (seed → ModelExpress → target)?
3. How is TP rank mismatch handled?
4. What happens if seed and target have different TP sizes?

**Recommendation**: Add tests for:
- TP rank matching logic
- Byte size validation
- Missing tensor handling
- ModelExpress server unavailability
- Timeout scenarios

### 8. Documentation

#### ⚠️ **Missing: Usage Documentation**

**Recommendation**: Add documentation explaining:
1. How to set up ModelExpress server
2. How to run seed instance with `--model-express-source`
3. How to run target instance with `--remote-instance-weight-loader-backend=model_express`
4. Model name coordination requirements
5. TP rank matching requirements

**Example**:
```markdown
## ModelExpress Remote Instance Loading

### Setup

1. Start ModelExpress server:
   ```bash
   modelexpress-server --port 8001
   ```

2. Start seed instance:
   ```bash
   python -m sglang.launch_server \
     --model-path meta-llama/Llama-3.1-70B \
     --model-express-url localhost:8001 \
     --model-express-model-name meta-llama/Llama-3.1-70B \
     --model-express-source \
     --remote-instance-weight-loader-start-seed-via-transfer-engine
   ```

3. Start target instance:
   ```bash
   python -m sglang.launch_server \
     --model-path meta-llama/Llama-3.1-70B \
     --load-format remote_instance \
     --remote-instance-weight-loader-backend model_express \
     --model-express-url localhost:8001 \
     --model-express-model-name meta-llama/Llama-3.1-70B
   ```

### Requirements

- Seed and target must have **matching TP sizes** (e.g., both TP=8)
- Each target TP rank connects to the corresponding seed TP rank
- ModelExpress server must be accessible from both instances
- TransferEngine must be initialized on both instances
```

## Specific PR Review Comments

### High Priority

1. **Dtype Inference**: Fix dtype mapping to use actual tensor dtypes instead of element size (see Section 2)
2. **NIXL Backend Support**: Add error handling for NIXL backend case (see Section 1)
3. **Validation**: Make CLI argument validation stricter (see Section 4)
4. **Model Name Default**: Improve model name default logic (see Section 4)

### Medium Priority

5. **Tensor Name Matching**: Add more robust tensor name matching with better error messages (see Section 3)
6. **Timeout Configuration**: Make timeout configurable and visible in errors (see Section 5)
7. **Memory Registration Timing**: Document/ensure weights are stable before registration (see Section 6)
8. **Documentation**: Add usage documentation (see Section 8)

### Low Priority

9. **Test Coverage**: Add comprehensive tests (see Section 7)
10. **Logging**: Add more detailed logging for debugging
11. **Error Recovery**: Consider retry logic for transient ModelExpress errors

## Alignment with ModelExpress PR 157

### ✅ **Correct Integration**

The SGLang PR correctly uses the `oneof` pattern from ModelExpress PR 157:
- Extracts `transfer_engine_session_id` from `backend_metadata` oneof
- Uses `WhichOneof()` to check backend type
- Provides appropriate error handling

### ⚠️ **Missing: Backend Selection**

The SGLang PR assumes TransferEngine backend. It doesn't:
- Check if source uses NIXL backend
- Provide fallback to NIXL if TransferEngine unavailable
- Allow configuration of preferred backend

**Recommendation**: Consider adding backend selection logic similar to what was discussed in ModelExpress PR 157 feedback.

## Conclusion

PR 19920 provides a solid integration of ModelExpress coordination for remote instance weight loading. The implementation correctly:

1. ✅ Uses the `oneof` pattern from ModelExpress PR 157
2. ✅ Implements TP rank matching
3. ✅ Handles byte-size matching for mixed-dtype models
4. ✅ Coordinates ready state via ModelExpress

**Key Improvements Needed**:
1. Fix dtype inference to use actual tensor dtypes
2. Add NIXL backend error handling
3. Improve validation and error messages
4. Add comprehensive documentation and tests

The PR is well-structured and follows SGLang's existing patterns. With the suggested improvements, it will provide a robust foundation for ModelExpress-coordinated weight loading.

---

## PR Review Comments

This section provides specific comments to make directly on PR 19920, organized by file and line numbers. These comments should be added as inline code review comments on the PR.

### File: `python/sglang/srt/model_loader/loader.py`

**Comment 1 - Line ~2340 (load_model_from_model_express, backend_field check)**
```
⚠️ Backend Type Handling: Add support for NIXL backend error case

Currently, the code only handles `transfer_engine_session_id` and raises a generic error for other backends. Consider adding explicit handling for NIXL:

```python
backend_field = source_worker.WhichOneof("backend_metadata")
if backend_field == "transfer_engine_session_id":
    seed_session_id = source_worker.transfer_engine_session_id
elif backend_field == "nixl_metadata":
    raise RuntimeError(
        f"ModelExpress: source worker {tp_rank} uses NIXL backend, "
        f"but MODEL_EXPRESS backend requires TransferEngine. "
        f"Please use a source with TransferEngine backend or use NIXL directly."
    )
else:
    raise RuntimeError(
        f"ModelExpress: unknown backend_metadata={backend_field} "
        f"for worker {tp_rank}. Expected 'transfer_engine_session_id'."
    )
```

This provides clearer error messages when backend types don't match.
```

**Comment 2 - Line ~2350 (tensor descriptor conversion)**
```
✅ Good: Byte size matching approach

The use of raw byte sizes (`td.size`) for matching is correct for RDMA transfers. RDMA is a memcpy operation, so byte-level matching is appropriate regardless of dtype differences (FP8 vs BF16, etc.).

Consider adding a comment explaining this:
```python
# Convert tensor descriptors to {name: (addr, size_bytes)} format
# Use raw byte sizes -- RDMA is a memcpy, dtype matching is not required
# The model's quantization logic handles dtype conversions, not the transfer layer
seed_weight_info = {}
```
```

**Comment 3 - Line ~2370 (tensor name matching)**
```
⚠️ Error Message Enhancement: Improve missing tensor error

When a tensor name is not found, provide more context:

```python
for name, tensor in model.named_parameters():
    weight_info = seed_weight_info.get(name, None)
    if weight_info is None:
        # Provide helpful context
        available_names = list(seed_weight_info.keys())
        logger.error(
            f"ModelExpress: tensor '{name}' not found in seed metadata. "
            f"Available tensors ({len(available_names)}): {available_names[:5]}..."
        )
        raise RuntimeError(
            f"ModelExpress: cannot find weight info for '{name}' "
            f"in seed metadata. This may indicate a model architecture mismatch "
            f"or different model versions between seed and target."
        )
```

This helps debug model architecture mismatches.
```

**Comment 4 - Line ~2280 (wait_for_ready call)**
```
⚠️ Timeout Configuration: Make timeout configurable

The `wait_for_ready` timeout is not visible in the code. Consider:

```python
timeout_seconds = getattr(load_config, 'model_express_ready_timeout', 7200)  # 2 hours default
ready, session_id, metadata_hash = mx_client.wait_for_ready(
    model_name, worker_id=tp_rank, timeout_seconds=timeout_seconds,
)
if not ready:
    raise RuntimeError(
        f"ModelExpress: timed out waiting for seed ready "
        f"(model={model_name}, worker={tp_rank}, timeout={timeout_seconds}s). "
        f"Check that seed instance is running and has published ready flag."
    )
```

Also consider adding `model_express_ready_timeout` to LoadConfig and ServerArgs.
```

### File: `python/sglang/srt/model_executor/model_runner.py`

**Comment 5 - Line ~700 (_publish_model_express_metadata, dtype inference)**
```
🔧 Critical: Fix dtype inference from element size

The current mapping is lossy and can misidentify dtypes:

```python
element_size_to_dtype = {1: "float8_e4m3fn", 2: "bfloat16", 4: "float32", 8: "float64"}
```

**Problem**: Multiple dtypes share the same element size:
- Size 2: `float16`, `bfloat16`, `int16`, `uint16`
- Size 1: `int8`, `uint8`, `float8_e4m3fn`, `float8_e5m2`

**Solution**: Use actual tensor dtype:

```python
tensors = []
for name, (addr, numel, element_size) in weight_info.items():
    # Get actual tensor to determine dtype
    param_dict = dict(self.model.named_parameters())
    if name not in param_dict:
        logger.warning(f"Parameter {name} not found in model, using element_size inference")
        dtype_str = element_size_to_dtype.get(element_size, "unknown")
    else:
        tensor = param_dict[name]
        dtype_str = str(tensor.dtype).replace("torch.", "")
    
    tensors.append(p2p_pb2.TensorDescriptor(
        name=name,
        addr=addr,
        size=numel * element_size,
        device_id=self.gpu_id,
        dtype=dtype_str,
    ))
```

**Alternative**: Modify `register_memory_region` to return dtype as well:
```python
# In remote_instance_weight_loader_utils.py
weight_info[name] = (addr, numel, element_size, str(tensor.dtype).replace("torch.", ""))
```
```

**Comment 6 - Line ~685 (model_name default)**
```
⚠️ Model Name Default: Improve consistency

Using `model_path` as default can lead to inconsistent model names:

```python
model_name = (
    self.server_args.model_express_model_name
    or self.server_args.model_path
)
```

**Issue**: `model_path` might be `/path/to/model` while target uses `meta-llama/Llama-3.1-70B`.

**Recommendation**:
```python
model_name = self.server_args.model_express_model_name
if not model_name:
    # Extract model name from model_path (last component)
    import os
    model_name = os.path.basename(self.server_args.model_path.rstrip('/'))
    logger.warning(
        f"ModelExpress: using model_name='{model_name}' from model_path. "
        f"Consider setting --model-express-model-name explicitly for consistency."
    )
```

Or require explicit model name:
```python
if not self.server_args.model_express_model_name:
    raise ValueError(
        "--model-express-model-name is required when using --model-express-source"
    )
```
```

**Comment 7 - Line ~1075 (model_specific_adjustment, memory registration timing)**
```
⚠️ Memory Registration Timing: Ensure weights are stable

The memory registration happens after model loading, but weights may be modified by `post_load_weights()`. Consider:

```python
# In model_specific_adjustment(), before ModelExpress publish:
# Ensure model weights are finalized (post_load_weights may modify weights)
if hasattr(self.model, "post_load_weights"):
    self.model.post_load_weights()

# Now register memory regions (weights are stable)
if self.server_args.model_express_source:
    if (
        self.remote_instance_transfer_engine_weight_info is None
        and self.remote_instance_transfer_engine is not None
    ):
        self.remote_instance_transfer_engine_weight_info = (
            register_memory_region(self.model, self.remote_instance_transfer_engine)
        )
    self._publish_model_express_metadata()
```

This ensures memory addresses remain valid after registration.
```

**Comment 8 - Line ~720 (publish_ready call)**
```
📝 Metadata Hash: Consider computing actual hash

Currently, `metadata_hash` is set to empty string:

```python
mx_client.publish_ready(
    model_name,
    worker_id=self.tp_rank,
    session_id=mx_client.session_id,
    metadata_hash="",  # Empty hash
)
```

Consider computing an actual hash of the tensor descriptors for validation:

```python
import hashlib
metadata_str = ",".join(sorted(f"{td.name}:{td.addr}:{td.size}" for td in tensors))
metadata_hash = hashlib.md5(metadata_str.encode()).hexdigest()

mx_client.publish_ready(
    model_name,
    worker_id=self.tp_rank,
    session_id=mx_client.session_id,
    metadata_hash=metadata_hash,
)
```

This enables target-side validation that metadata hasn't changed.
```

### File: `python/sglang/srt/server_args.py`

**Comment 9 - Line ~2722 (validation logic)**
```
⚠️ Validation: Make validation stricter

The current validation silently falls back to `auto`:

```python
if self.remote_instance_weight_loader_backend == "model_express":
    if self.model_express_url is None:
        logger.warning("Fallback load_format to 'auto'...")
        self.load_format = "auto"
```

**Recommendation**: Raise an error instead:

```python
if self.remote_instance_weight_loader_backend == "model_express":
    if self.model_express_url is None:
        raise ValueError(
            "--model-express-url is required when using "
            "--remote-instance-weight-loader-backend=model_express"
        )
    if not self.validate_transfer_engine():
        raise ValueError(
            "TransferEngine is required for model_express backend. "
            "Please install mooncake.engine or use a different backend."
        )
```

Silent fallback can lead to confusion when users expect model_express backend.
```

**Comment 10 - Line ~5235 (CLI argument help text)**
```
📝 Documentation: Enhance help text

The help text for `--model-express-source` could be more descriptive:

```python
parser.add_argument(
    "--model-express-source",
    action="store_true",
    help=(
        "Run as a ModelExpress seed source: publish TransferEngine metadata "
        "to the ModelExpress server after loading weights. "
        "Requires --model-express-url and TransferEngine initialization. "
        "Target instances can then load weights via --remote-instance-weight-loader-backend=model_express."
    ),
)
```

This clarifies the relationship between source and target modes.
```

**Comment 11 - Line ~5783 (validate_transfer_engine, ModelExpress source check)**
```
✅ Good: TransferEngine validation includes ModelExpress source

The validation correctly checks for ModelExpress source mode:

```python
if self.model_express_source:
    return True
```

This ensures TransferEngine is initialized when running as a seed source.
```

### File: `python/sglang/srt/configs/load_config.py`

**Comment 12 - Line ~78-79 (LoadConfig fields)**
```
✅ Good: Clean addition of ModelExpress fields

The addition of `model_express_url` and `model_express_model_name` to LoadConfig is clean and follows existing patterns.

Consider adding a comment:
```python
# ModelExpress coordination fields (for remote_instance_weight_loader_backend=model_express)
model_express_url: Optional[str] = None
model_express_model_name: Optional[str] = None
```
```

### Testing & Documentation

**Comment 13 - Missing: Test Coverage**
```
✅ Test Coverage Needed

Please add tests for:
1. **TP rank matching**: Verify each target rank connects to correct seed rank
2. **Byte size validation**: Test size mismatch detection
3. **Missing tensor handling**: Test behavior when tensor names don't match
4. **ModelExpress server unavailability**: Test error handling
5. **Timeout scenarios**: Test ready state timeout handling
6. **Mixed dtype models**: Test FP8 + BF16 models

Example test structure:
```python
def test_model_express_tp_rank_matching():
    # Test that target TP rank 0 connects to seed TP rank 0
    ...

def test_model_express_byte_size_validation():
    # Test that size mismatches are detected
    ...
```
```

**Comment 14 - Missing: Usage Documentation**
```
📚 Documentation Needed

Please add documentation explaining:
1. How to set up ModelExpress server
2. How to run seed instance with `--model-express-source`
3. How to run target instance with `--remote-instance-weight-loader-backend=model_express`
4. Model name coordination requirements
5. TP rank matching requirements (seed and target must have same TP size)

Consider adding to `docs/advanced_features/rfork.md` or creating a new section.
```

### Summary of Priority Comments

**High Priority (Must Address)**:
- Comment 5: Fix dtype inference from element size (critical for correctness)
- Comment 9: Make validation stricter (prevents silent failures)
- Comment 6: Improve model name default logic (prevents coordination failures)

**Medium Priority (Should Address)**:
- Comment 1: Add NIXL backend error handling
- Comment 3: Improve missing tensor error messages
- Comment 4: Make timeout configurable
- Comment 7: Ensure weights are stable before registration
- Comment 13: Add test coverage

**Low Priority (Nice to Have)**:
- Comment 2: Add comment explaining byte size matching
- Comment 8: Compute actual metadata hash
- Comment 10: Enhance help text
- Comment 12: Add comments to LoadConfig
- Comment 14: Add usage documentation
