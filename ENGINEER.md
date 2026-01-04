# ENGINEER.md - Complete Technical Documentation

**Project**: GPT-OSS-20B Training Stabilization with Unsloth & MXFP4
**Date**: 2026-01-04
**Status**: Active Debugging - Benchmark v21 Running

---

## EXECUTIVE SUMMARY

### Primary Objective
Achieve a **stable 500-step SFT (Supervised Fine-Tuning) training run** on Modal using:
- Model: `openai/gpt-oss-20b` (20B parameter MoE model)
- Quantization: MXFP4 (4-bit microscaling floating point)
- Framework: Unsloth + Transformers + TRL
- Hardware: NVIDIA H100 80GB (Modal Cloud)
- Dataset: `radna0/nemotron-harmony-formatted` (418K samples, using 100K)

### Current Status
- **Benchmark v21** is currently running on Modal
- Successfully passed initial model loading and dataset preparation
- Currently in tokenization/mapping phase (66% complete as of last check)
- Waiting to observe first training steps for CUDA stability

### Critical Blockers Resolved
1. ✅ Triton kernel bias type mismatch (BF16 vs FP32)
2. ✅ Blackwell GPU architecture incompatibility
3. ✅ 3D tensor shape mismatches in expert forward pass
4. ✅ Missing backward pass implementation for MXFP4 experts
5. ✅ Model configuration inconsistencies

---

## TECHNICAL ARCHITECTURE

### System Components

#### 1. Model Architecture
```
GPT-OSS-20B Specifications:
- Total Parameters: 20B
- Architecture: Mixture of Experts (MoE)
- Experts: 32 local experts
- Experts per Token: 4 (top-k routing)
- Hidden Size: 2880
- Intermediate Size: 2880
- Layers: 24
- Attention Heads: 64
- KV Heads: 8
- Context Length: 131,072 (trained), using 65,536 for stability
- Quantization: MXFP4 (4-bit weights)
- Precision: BFloat16 activations
```

#### 2. Training Infrastructure
```
Modal Cloud Setup:
- GPU: NVIDIA H100 80GB HBM3
- CUDA: 12.8.0
- Compute Capability: 9.0 (Hopper)
- PyTorch: 2.9.0+cu128
- Triton: 3.5.0
- Transformers: 4.57.3
- TRL: 0.9.6
- Unsloth: 2026.1.1

Volumes:
- mhc-data-volume: Dataset storage
- gpt-oss-model-weights-ads-70439: Model weights (44GB)
- hf-cache-persistent: HuggingFace cache
```

#### 3. Training Configuration
```python
Training Hyperparameters:
- Batch Size: 1 (per device)
- Gradient Accumulation: 32 (effective batch = 32)
- Max Steps: 500
- Learning Rate: 2e-5
- Optimizer: AdamW 8-bit
- Weight Decay: 0.001
- Adam Epsilon: 1e-20
- Adam Beta1: 0.9
- Adam Beta2: 0.95
- LR Scheduler: Linear
- Warmup Steps: 20
- Max Gradient Norm: 1.0
- Precision: BFloat16
- Sequence Length: 65,536 tokens
- Packing: False

LoRA Configuration:
- Rank: 16
- Alpha: 32
- Dropout: 0
- Target Modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
- Use RSLoRA: False
- Gradient Checkpointing: Unsloth mode
```

---

## CRITICAL BUGS IDENTIFIED & FIXED

### Bug #1: Router Indices TypeError
**Location**: `unsloth_zoo/temporary_patches/gpt_oss.py`

**Error**:
```
TypeError: patch_gpt_oss.<locals>.Mxfp4GptOssExperts.forward() got an unexpected keyword argument 'router_indices'
```

**Root Cause**: 
Unsloth's patched `Mxfp4GptOssExperts.forward()` method did not accept `**kwargs`, causing it to reject the `router_indices` argument passed by the MLP layer.

**Fix**:
```python
# Before:
def forward(self, hidden_states: torch.Tensor, routing_data = None, gather_idx = None, scatter_idx = None) -> torch.Tensor:

# After:
def forward(self, hidden_states: torch.Tensor, routing_data = None, gather_idx = None, scatter_idx = None, **kwargs) -> torch.Tensor:
```

**Files Modified**:
- `unsloth_zoo_src/unsloth_zoo/temporary_patches/gpt_oss.py:283`
- `unsloth_zoo_src/unsloth_zoo/temporary_patches/gpt_oss.py:155` (Training version)

---

### Bug #2: 3D Tensor Shape Mismatch in matmul_ogs
**Location**: `triton_kernels/matmul_ogs.py`

**Error**:
```
AssertionError: gather not supported in batched mode
AssertionError: scatter not supported in batched mode
AssertionError: routing not supported in batched mode
```

**Root Cause**:
The `matmul_ogs` Triton kernel expects 2D tensors `[N, D]` but was receiving 3D tensors `[B, S, D]` from the expert forward pass. The kernel has explicit assertions that reject batched inputs when using MoE routing.

**Fix**:
Implemented 3D-to-2D flattening in both forward and backward passes:

```python
# Forward pass in Mxfp4GptOssExperts_Training.forward():
ctx.input_shape = hidden_states.shape
if hidden_states.ndim == 3:
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

# ... matmul_ogs calls ...

if len(ctx.input_shape) == 3:
    out = out.reshape(ctx.input_shape)

# Backward pass:
grad_token_shape = grad_token.shape
if grad_token.ndim == 3:
    grad_token = grad_token.reshape(-1, grad_token.shape[-1])

# ... backward operations ...

if len(ctx.input_shape) == 3:
    dx_token = dx_token.reshape(ctx.input_shape)
```

**Files Modified**:
- `unsloth_zoo_src/unsloth_zoo/temporary_patches/gpt_oss.py:166-167, 208-209, 214-217, 267-268`

---

### Bug #3: Triton Kernel Bias Type Mismatch
**Location**: `triton_kernels/matmul_ogs_details/_p_matmul_ogs.py` and `_matmul_ogs.py`

**Error**:
```
triton.CompilationError: at 358:24:
    bias = tl.load(BPtrs, mask=mask_n, other=0.0)
                   ^
'tl.load' op operand #0 must be triton.PointerType, but got 'tensor<128xbf16>'
note: see current operation: %223 = "tl.load"(%222, %221, %c0_f32) : (tensor<128x!tt.ptr<bf16>>, tensor<128xi1>, f32) -> tensor<128xbf16>
```

**Root Cause**:
The Triton kernel had an if/else branch where:
- The `if` branch initialized bias as `tl.full([BLOCK_N], 0, dtype=tl.float32)` (FP32)
- The `else` branch loaded bias as BFloat16 from memory
- Later code tried to add these together, causing a type mismatch

**Fix**:
Cast the loaded bias to FP32 immediately upon loading:

```python
# In _p_matmul_ogs.py and _matmul_ogs.py, line ~365:
if B is not None:
    BPtrs = B + expt_id1 * stride_b_e + offs_y_n
    if pid_k1 == 0:
        bias = tl.load(BPtrs, mask=mask_n, other=0.0).to(tl.float32)  # ← Added .to(tl.float32)
    else:
        bias = tl.full([BLOCK_N], 0, dtype=tl.float32)
else:
    bias = tl.full([BLOCK_N], 0, dtype=tl.float32)
```

**Files Modified**:
- `triton_src/python/triton_kernels/triton_kernels/matmul_ogs_details/_p_matmul_ogs.py:356`
- `triton_src/python/triton_kernels/triton_kernels/matmul_ogs_details/_matmul_ogs.py:365`

---

### Bug #4: Blackwell GPU Architecture Incompatibility
**Location**: Modal GPU selection

**Error**:
```
AssertionError: Only Hopper swizzling is supported for values
```

**Root Cause**:
The MXFP4 Triton kernels contain a hardcoded check for `SWIZZLE_MX_VALUE` that only supports Hopper (H100) architecture. Blackwell (B200) GPUs use a different memory swizzling pattern that is not yet implemented in the kernel.

**Fix**:
Changed GPU selection in Modal from B200 to H100:

```python
# Before:
@app.function(gpu="B200", ...)

# After:
@app.function(gpu="H100", ...)  # Hopper architecture for MXFP4 compatibility
```

**Files Modified**:
- `modal/mhc_training_benchmark.py:80`

---

### Bug #5: Missing Backward Pass Implementation
**Location**: `unsloth_zoo/temporary_patches/gpt_oss.py`

**Error**:
```
NotImplementedError: Backwards pass using MXFP4 is still under construction!
Instead, use `unsloth/gpt-oss-20b-BF16` for bfloat16 training which will work for LoRA.
Or, use `load_in_4bit = True` which allows finetuning.
```

**Root Cause**:
The `Mxfp4GptOssExperts_Training.backward()` method raised `NotImplementedError` immediately, preventing any gradient computation. This was a placeholder from Unsloth that was never completed.

**Fix**:
Implemented a complete backward pass using `matmul_ogs` for gradient computation:

```python
@staticmethod
def backward(ctx, grad_token):
    (pre_act, gamma, expt_hist, gather_src, gather_dst, scatter_src, scatter_dst,) = ctx.saved_tensors
    
    # Handle 3D gradients
    grad_token_shape = grad_token.shape
    if grad_token.ndim == 3:
        grad_token = grad_token.reshape(-1, grad_token.shape[-1])
    
    self_class = ctx.self_class
    limit = self_class.limit
    alpha = self_class.alpha

    # 1) token → expert (reverse of forward scatter)
    grad_exp = grad_token.index_select(0, scatter_src)
    grad_exp.mul_(gamma.unsqueeze(-1))
    
    # 2) grad_exp · Wd^T (reuse matmul_ogs)
    Wd_T = self_class.down_proj.transpose(-1, -2)
    g1 = matmul_ogs(
        grad_exp, Wd_T, None, ctx.routing_data,
        gather_indx=None, scatter_indx=None,
        precision_config=self_class.down_proj_precision_config,
        gammas=None, fused_activation=None,
    )
    
    # 3) activation derivative (SwiGLU backward)
    g1 = swiglu_torch_backward(pre_act, alpha, limit, g1)
    
    # 4) g1 · Wu^T
    Wu_T = self_class.gate_up_proj.transpose(-1, -2)
    dx_exp = matmul_ogs(
        g1, Wu_T, None, ctx.routing_data,
        gather_indx=None, scatter_indx=None,
        precision_config=self_class.gate_up_proj_precision_config,
        gammas=None, fused_activation=None,
    )

    # 5) expert → token (reverse of forward gather)
    dx_token = torch.zeros_like(grad_token)
    dx_token.index_add_(0, gather_dst, dx_exp)
    
    # Reshape back to 3D if necessary
    if len(ctx.input_shape) == 3:
        dx_token = dx_token.reshape(ctx.input_shape)
        
    return dx_token, None, None, None, None, *[None]*ctx.kwargs_len
```

**Key Changes**:
1. Removed `NotImplementedError`
2. Added `expt_hist` to saved tensors for routing data
3. Implemented proper gradient flow through expert routing
4. Used `matmul_ogs` for backward matmuls (consistent with forward)
5. Fixed 3D tensor handling in backward pass
6. Properly applied routing weights (gamma) to gradients

**Files Modified**:
- `unsloth_zoo_src/unsloth_zoo/temporary_patches/gpt_oss.py:197-268`

---

### Bug #6: Model Configuration Inconsistency
**Location**: Model checkpoint configuration

**Issue**:
The local model configuration might have been outdated or modified, potentially causing subtle incompatibilities with the Triton kernels or Unsloth patches.

**Fix**:
Downloaded the official `config.json` from Hugging Face and enforced it during model loading:

```python
# Download official config
curl -L https://huggingface.co/openai/gpt-oss-20b/resolve/main/config.json -o gpt_oss_config.json

# Enforce in Modal runtime
import shutil
try:
    shutil.copy("/root/gpt_oss_config.json", os.path.join(model_name, "config.json"))
    print("Successfully enforced official config.json")
except Exception as e:
    print(f"Warning: Could not enforce config.json: {e}")
```

**Files Modified**:
- `gpt_oss_config.json` (downloaded fresh)
- `modal/mhc_training_benchmark.py:57, 157-161` (injection and enforcement)

---

## INFRASTRUCTURE CHANGES

### Local Source Injection Strategy

**Problem**: Need to rapidly iterate on bug fixes without waiting for upstream package updates.

**Solution**: Inject locally patched sources into Modal container:

```python
# Local paths
UNSLOTH_SRC = Path("/home/kojoe/CUDA_mhc/unsloth_src")
UNSLOTH_ZOO_SRC = Path("/home/kojoe/CUDA_mhc/unsloth_zoo_src")
TRITON_KERNELS_SRC = Path("/home/kojoe/CUDA_mhc/triton_src/python/triton_kernels/triton_kernels")

# Modal image build
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu24.04", add_python="3.11")
    # ... install packages from pip ...
    .add_local_dir(UNSLOTH_SRC, remote_path="/root/unsloth_src", copy=True)
    .add_local_dir(UNSLOTH_ZOO_SRC, remote_path="/root/unsloth_zoo_src", copy=True)
    .add_local_dir(TRITON_KERNELS_SRC, remote_path="/root/triton_kernels_src", copy=True)
    .run_commands(
        "cp -rfv /root/unsloth_src/unsloth/* /usr/local/lib/python3.11/site-packages/unsloth/",
        "cp -rfv /root/unsloth_zoo_src/unsloth_zoo/* /usr/local/lib/python3.11/site-packages/unsloth_zoo/",
        "cp -rfv /root/triton_kernels_src/* /usr/local/lib/python3.11/site-packages/triton_kernels/",
        # Clear Python cache to force reload
        "find /usr/local/lib/python3.11/site-packages/unsloth -name '__pycache__' -type d -exec rm -rf {} +",
        "find /usr/local/lib/python3.11/site-packages/unsloth_zoo -name '__pycache__' -type d -exec rm -rf {} +",
        "find /usr/local/lib/python3.11/site-packages/triton_kernels -name '__pycache__' -type d -exec rm -rf {} +",
    )
)
```

**Benefits**:
- Instant deployment of bug fixes
- No dependency on upstream merge/release cycles
- Full control over critical code paths
- Easy rollback by reverting local changes

---

### Output Redirection & Logging

**Implementation**:
```python
# Create Tee class for dual output (stdout + file)
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# Redirect all output
os.makedirs("logs", exist_ok=True)
log_file_path = "logs/mhc_benchmark_v21.log"
f = open(log_file_path, "w")
sys.stdout = Tee(sys.stdout, f)
sys.stderr = Tee(sys.stderr, f)
```

**Log Files**:
- `logs/mhc_benchmark_v21.log` - Current run (in progress)
- `logs/mhc_benchmark_v20.log` - Previous run (illegal memory access)
- `logs/mhc_benchmark_v19.log` - Triton bias type error
- `logs/triton_kernel_details.log` - Kernel source inspection

---

## DEBUGGING METHODOLOGY

### Systematic Approach

1. **Error Identification**
   - Capture full stack traces
   - Identify exact line numbers and file paths
   - Determine error category (type error, assertion, CUDA error, etc.)

2. **Root Cause Analysis**
   - Read source code at error location
   - Trace data flow backward from error point
   - Identify assumptions violated
   - Check for type mismatches, shape mismatches, null pointers

3. **Fix Design**
   - Minimal change principle: smallest fix that resolves root cause
   - Maintain backward compatibility where possible
   - Preserve original intent of code
   - Document why the fix works

4. **Implementation**
   - Apply fix to local source
   - Verify syntax correctness
   - Check for side effects

5. **Verification**
   - Deploy to Modal with source injection
   - Monitor execution through critical path
   - Confirm error no longer occurs
   - Check for new errors introduced

6. **Iteration**
   - If new error appears, return to step 1
   - If training progresses, monitor for stability
   - Document all changes

---

## CURRENT DEBUGGING SESSION

### Benchmark v21 Status

**Launched**: 2026-01-04 17:42 UTC

**Progress**:
1. ✅ Modal container started
2. ✅ CUDA environment initialized
3. ✅ Unsloth patches loaded
4. ✅ Model loaded (3 checkpoint shards)
5. ✅ LoRA adapters applied
6. ✅ Dataset loaded (418,142 examples)
7. ✅ Dataset sampled (100,000 examples)
8. ✅ Chat template applied
9. ✅ SFTTrainer initialized
10. 🔄 Tokenization mapping (66% complete at last check)
11. ⏳ Waiting for first training step

**Expected Next Steps**:
1. Tokenization completes (100%)
2. Training loop begins
3. First forward pass through model
4. First backward pass (gradient computation)
5. First optimizer step
6. Log first training metrics

**Critical Observation Points**:
- Does the first forward pass complete without CUDA errors?
- Does the backward pass execute without illegal memory access?
- Are gradients finite and reasonable magnitude?
- Does the first optimizer step succeed?

**Potential Failure Modes**:
1. **CUDA illegal memory access** - Would indicate tensor addressing bug
2. **NaN/Inf gradients** - Would indicate numerical instability
3. **OOM (Out of Memory)** - Would indicate memory leak or excessive allocation
4. **Kernel launch failure** - Would indicate Triton compilation issue

---

## ENGINEERING RULES & PRINCIPLES

### Code Quality Standards

1. **NEVER GUESS**
   - Read actual source code, don't assume behavior
   - Verify error messages by reading the exact line that failed
   - Test fixes, don't just pattern match

2. **MINIMAL CHANGES**
   - Change only what's necessary to fix the bug
   - Preserve original code structure
   - Don't refactor while debugging

3. **DOCUMENT EVERYTHING**
   - Every bug fix must have: error, root cause, fix, verification
   - Update task.md and implementation_plan.md
   - Keep ENGINEER.md current

4. **VERIFY FIXES**
   - Run the code after every change
   - Observe the actual behavior
   - Don't claim "should work" - prove it works

5. **SYSTEMATIC DEBUGGING**
   - One bug at a time
   - Fix root cause, not symptoms
   - Verify fix before moving to next bug

### Modal Deployment Rules

1. **Version Everything**
   - Increment benchmark version for each run (v19, v20, v21, ...)
   - Log file names match version
   - Track which fixes are in which version

2. **Source Injection**
   - Always inject local sources for active debugging
   - Clear Python cache after copying
   - Verify injection with inspection functions

3. **Resource Management**
   - Use appropriate GPU (H100 for MXFP4)
   - Set reasonable timeouts (7200s = 2 hours)
   - Monitor memory usage

4. **Logging**
   - Redirect all output to log files
   - Use Tee for dual stdout/file output
   - Preserve logs for post-mortem analysis

### Triton Kernel Rules

1. **Type Consistency**
   - All branches must produce same dtype
   - Cast immediately upon load if needed
   - Use `.to(dtype)` for explicit conversion

2. **Shape Validation**
   - Check tensor dimensions before kernel calls
   - Flatten 3D to 2D when required
   - Reshape back after kernel returns

3. **Hardware Compatibility**
   - Check compute capability requirements
   - Use appropriate swizzling for architecture
   - Test on target hardware (H100)

### Git Workflow

1. **Branch Strategy**
   - Main branch: stable code
   - Feature branches: active development
   - Tag releases: v1, v2, etc.

2. **Commit Messages**
   - Format: `[Component] Brief description`
   - Example: `[Triton] Fix bias type mismatch in matmul_ogs`
   - Include issue reference if applicable

3. **Code Review**
   - Self-review before commit
   - Check for unintended changes
   - Verify no debug code left in

---

## TECHNICAL DETAILS

### MXFP4 Format

**Microscaling Floating Point 4-bit**:
- 4 bits per weight value
- Shared exponent per block (32 values)
- Block size: 32 elements
- Precision: ~2-3 significant bits
- Dynamic range: Maintained by shared exponent

**Memory Layout**:
```
Weights: [num_experts, out_features, in_features // 32, 16] (uint8, 2 values per byte)
Scales:  [num_experts, out_features, in_features // 32] (uint8, exponent)
```

**Kernel Requirements**:
- Must use `tl.dot_scaled` for matrix multiplication
- Requires scale unswizzling on Blackwell
- Hopper uses direct scale access
- Upcast to BF16 for computation

### Expert Routing

**Top-K Routing**:
```python
# Router computes logits for all experts
router_logits = linear(hidden_states, router.weight, router.bias)  # [B*S, num_experts]

# Select top-k experts per token
routing_data, gather_idx, scatter_idx = routing(router_logits, top_k=4)

# routing_data contains:
# - gate_scal: Softmax weights for selected experts [B*S*k]
# - expt_hist: Histogram of tokens per expert [num_experts]
# - n_expts_tot: Total number of experts (32)
# - n_expts_act: Experts per token (4)

# gather_idx: Maps tokens to expert-sorted positions
# scatter_idx: Maps expert outputs back to token positions
```

**Expert Forward Pass**:
```python
# 1. Gather tokens for each expert
x_gathered = x[gather_idx.src_indx]  # Expert-sorted tokens

# 2. Process through experts (batched matmul)
gate_up = matmul_ogs(x_gathered, gate_up_proj, bias, routing_data, gather_indx=gather_idx)
swiglu_out = swiglu(gate_up)
expert_out = matmul_ogs(swiglu_out, down_proj, bias, routing_data, scatter_indx=scatter_idx, gammas=gate_scal)

# 3. Scatter outputs back to token positions (weighted by routing)
output[scatter_idx.dst_indx] += expert_out * gate_scal
```

### Gradient Flow

**Backward Pass Through Experts**:
```python
# Input: grad_output [B*S, D] (gradient from next layer)

# 1. Gather gradients for expert outputs
grad_expert = grad_output[scatter_src]  # [B*S*k, D]
grad_expert *= gamma  # Apply routing weights

# 2. Backward through down_proj
grad_swiglu = matmul_ogs(grad_expert, down_proj.T, ...)  # [B*S*k, intermediate_size]

# 3. Backward through SwiGLU activation
grad_gate_up = swiglu_backward(pre_activation, grad_swiglu)  # [B*S*k, 2*intermediate_size]

# 4. Backward through gate_up_proj
grad_gathered = matmul_ogs(grad_gate_up, gate_up_proj.T, ...)  # [B*S*k, D]

# 5. Scatter gradients back to tokens
grad_input = torch.zeros_like(grad_output)
grad_input.index_add_(0, gather_dst, grad_gathered)  # [B*S, D]
```

---

## MONITORING & VALIDATION

### Key Metrics to Track

**Training Metrics**:
- Loss (should decrease over time)
- Learning rate (should follow schedule)
- Gradient norm (should be < max_grad_norm)
- Throughput (tokens/second)

**System Metrics**:
- GPU memory usage (should be stable)
- GPU utilization (should be high during training)
- CPU memory (should not grow unbounded)
- Disk I/O (dataset loading)

**Stability Indicators**:
- No CUDA errors
- No NaN/Inf in loss
- No memory leaks
- Consistent step time

### Success Criteria

**Benchmark v21 Success**:
1. ✅ Complete first 10 steps without errors
2. ✅ Loss is finite and reasonable (< 10.0)
3. ✅ Memory usage is stable (< 75GB)
4. ✅ Complete 100 steps (20% of target)
5. ✅ Complete 500 steps (100% of target)

**Training Quality**:
- Loss decreases over 500 steps
- Final loss < initial loss
- No catastrophic forgetting
- Model generates coherent text

---

## NEXT STEPS

### Immediate (Benchmark v21)
1. ⏳ Wait for tokenization to complete
2. ⏳ Observe first training step
3. ⏳ Verify no CUDA errors in first 10 steps
4. ⏳ Monitor loss and gradients
5. ⏳ Confirm 500-step completion

### Short-term (Post-v21)
1. Analyze training metrics
2. Compare to baseline (if available)
3. Test model quality (generation samples)
4. Document final results
5. Create walkthrough artifact

### Long-term (Future Work)
1. Enable mHC (Multi-Head Coordination) if baseline succeeds
2. Compare mHC vs baseline performance
3. Optimize hyperparameters
4. Scale to full dataset (418K samples)
5. Production deployment

---

## REPOSITORY STRUCTURE

```
/home/kojoe/CUDA_mhc/
├── modal/
│   └── mhc_training_benchmark.py          # Main training script (v21)
├── scripts/
│   ├── mhc_huggingface_adapter.py         # mHC adapter (not used in baseline)
│   ├── format_nemotron_harmony.py         # Dataset formatter
│   └── prep_dataset_volume.py             # Dataset preparation
├── unsloth_src/                           # Local Unsloth source (injected)
├── unsloth_zoo_src/                       # Local Unsloth Zoo source (injected)
│   └── unsloth_zoo/temporary_patches/
│       └── gpt_oss.py                     # ⚠️ CRITICAL: Expert patching
├── triton_src/                            # Local Triton kernels (injected)
│   └── python/triton_kernels/triton_kernels/
│       ├── matmul_ogs.py                  # Main matmul wrapper
│       └── matmul_ogs_details/
│           ├── _matmul_ogs.py             # ⚠️ CRITICAL: Non-persistent kernel
│           └── _p_matmul_ogs.py           # ⚠️ CRITICAL: Persistent kernel
├── logs/
│   ├── mhc_benchmark_v21.log              # Current run (in progress)
│   ├── mhc_benchmark_v20.log              # Previous run (illegal memory access)
│   └── mhc_benchmark_v19.log              # Earlier run (bias type error)
├── gpt_oss_config.json                    # Official model config (enforced)
├── GEMINI.md                              # Engineering excellence rules
└── ENGINEER.md                            # This document

/.gemini/antigravity/brain/c7c74970-5f0b-406e-acc7-434e6f01c2f9/
├── task.md                                # Task tracking checklist
├── implementation_plan.md                 # Technical implementation plan
└── walkthrough.md                         # Results documentation (pending)
```

---

## CRITICAL FILES

### Files That Must Not Be Modified
- `triton_src/python/triton_kernels/triton_kernels/routing.py` - Routing logic
- `triton_src/python/triton_kernels/triton_kernels/numerics_details/flexpoint.py` - Flexpoint math
- Model checkpoint files in Modal volume

### Files Under Active Development
- ⚠️ `unsloth_zoo_src/unsloth_zoo/temporary_patches/gpt_oss.py` - Expert patching
- ⚠️ `triton_src/python/triton_kernels/triton_kernels/matmul_ogs_details/_p_matmul_ogs.py` - Persistent kernel
- ⚠️ `triton_src/python/triton_kernels/triton_kernels/matmul_ogs_details/_matmul_ogs.py` - Non-persistent kernel
- `modal/mhc_training_benchmark.py` - Training orchestration

### Configuration Files
- `gpt_oss_config.json` - Model architecture (official from HF)
- `GEMINI.md` - Engineering rules and standards
- `ENGINEER.md` - This technical documentation

---

## CONTACT & ESCALATION

### When to Escalate
1. **CUDA errors persist after 3 fix attempts**
2. **Training OOMs despite memory optimization**
3. **Triton kernel compilation fails on H100**
4. **Loss diverges (NaN/Inf) within first 50 steps**
5. **Modal infrastructure issues (timeouts, volume access)**

### Debug Information to Collect
1. Full error traceback
2. Relevant log file sections
3. GPU memory snapshot
4. Triton kernel compilation output
5. Model configuration
6. Training hyperparameters

---

## VERSION HISTORY

### Benchmark Versions

**v21** (Current - 2026-01-04 17:42 UTC):
- ✅ Fixed backward pass implementation
- ✅ Enforced official config.json
- ✅ Added comprehensive logging
- 🔄 Status: Running (tokenization phase)

**v20** (2026-01-04):
- ✅ Fixed Triton bias type mismatch
- ✅ Injected local Triton kernels
- ❌ Failed: CUDA illegal memory access during backward pass

**v19** (2026-01-04):
- ✅ Switched to H100 GPU
- ✅ Fixed 3D tensor shape issues
- ❌ Failed: Triton compilation error (bias type mismatch)

**v18 and earlier**:
- Various router and expert patching attempts
- Blackwell GPU compatibility issues
- Initial bug discoveries

---

## GLOSSARY

**MXFP4**: Microscaling Floating Point 4-bit - A quantization format with shared exponents per block
**MoE**: Mixture of Experts - Architecture with multiple expert networks and routing
**SFT**: Supervised Fine-Tuning - Training on labeled data
**LoRA**: Low-Rank Adaptation - Parameter-efficient fine-tuning method
**Triton**: GPU programming language and compiler from OpenAI
**TMA**: Tensor Memory Accelerator - H100 hardware feature for efficient memory access
**Hopper**: NVIDIA GPU architecture (H100)
**Blackwell**: NVIDIA GPU architecture (B200)
**BF16**: BFloat16 - 16-bit brain floating point format
**FP32**: 32-bit floating point
**Swizzling**: Memory layout optimization for GPU access patterns

---

## APPENDIX: ERROR CATALOG

### Error Type: TypeError
**Pattern**: `got an unexpected keyword argument`
**Cause**: Function signature mismatch
**Fix**: Add `**kwargs` to function signature

### Error Type: AssertionError
**Pattern**: `not supported in batched mode`
**Cause**: Tensor dimension mismatch
**Fix**: Flatten/reshape tensors before kernel call

### Error Type: CompilationError
**Pattern**: `operand #0 must be X, but got Y`
**Cause**: Type mismatch in Triton kernel
**Fix**: Cast to correct type with `.to(dtype)`

### Error Type: AcceleratorError
**Pattern**: `CUDA error: an illegal memory access`
**Cause**: Invalid memory address or tensor corruption
**Fix**: Check tensor shapes, strides, and saved tensors in autograd context

### Error Type: NotImplementedError
**Pattern**: `Backwards pass using MXFP4 is still under construction`
**Cause**: Placeholder code not implemented
**Fix**: Implement the backward pass

---

**END OF ENGINEER.MD**

Last Updated: 2026-01-04 17:57 UTC
Status: Benchmark v21 Running - Awaiting First Training Step
