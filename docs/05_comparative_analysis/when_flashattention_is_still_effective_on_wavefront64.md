# When FlashAttention Is Still Effective on Wavefront-64 GPUs

This document complements:

- `attention_kernel_contract_for_wavefront64.md`
- `where_flashattention_violates_wavefront64_contract.md`

by identifying **specific architectural regimes where FlashAttention remains a valid
and sometimes efficient design choice** on AMD wavefront-64 GPUs.

The goal is not to defend FlashAttention, but to apply **architectural judgment**.

---

## 1. Framing the Question Correctly

FlashAttention is often discussed as a binary choice:

- “Good” on NVIDIA
- “Bad” on AMD

This framing is incorrect.

FlashAttention’s performance on wavefront-64 GPUs depends on **how strongly its design
violates the kernel contract**, which varies across:

- Sequence length
- Tile size
- Head dimension
- Compute vs bandwidth balance
- Occupancy regime

The correct question is not:

> “Is FlashAttention optimal on wavefront-64?”

But:

> “Under what conditions do its assumptions still hold well enough?”

---

## 2. Key Observation

FlashAttention’s architectural violations are **amplified by scale**.

When scale is reduced, many violations become **tolerable rather than dominant**.

Violations compound only when **multiple stressors coincide simultaneously**.

---

## 3. Case 1 — Short Sequence Lengths

### Condition
- Sequence length is small (e.g., ≤256–512 tokens)
- KV loop iteration count is low

### Why It Works

- Accumulator lifetime is short
- Online softmax state does not persist long enough to dominate VGPR usage
- AccVGPR ↔ ArchVGPR domain crossings are amortized over few iterations

### Architectural Outcome

- Occupancy remains ≥2 waves per SIMD
- MFMA pipelines remain fed
- Register pressure does not reach cliff thresholds

### Verdict

✅ **FlashAttention is effective**

In this regime, FlashAttention behaves similarly to a conventional fused attention kernel.

---

## 4. Case 2 — Smaller Tile Sizes

### Condition
- Reduced tile dimensions (e.g., `BLOCK_M=64` instead of 128)
- Smaller accumulator footprint

### Why It Works

- AccVGPR usage per wavefront is reduced
- Register pressure grows more gradually
- Occupancy degradation becomes smoother rather than step-function collapse

### Tradeoff

- Lower arithmetic intensity
- Increased loop overhead

### Architectural Outcome

- Improved wave residency
- Better latency hiding

### Verdict

✅ **FlashAttention remains competitive**, especially when bandwidth is available.

---

## 5. Case 3 — Smaller Head Dimensions

### Condition
- Head dimension is modest (e.g., ≤64)

### Why It Works

- Accumulator footprint in AccVGPRs is reduced
- Softmax state consumes fewer ArchVGPRs
- Domain crossing cost is reduced due to lower data volume

### Architectural Outcome

- Occupancy cliffs are less severe
- MFMA utilization remains reasonable

### Verdict

✅ **Acceptable with careful tiling**

---

## 6. Case 4 — Compute-Bound Attention Regimes

### Condition
- High arithmetic intensity
- Large head dimensions
- MFMA utilization dominates execution time

### Why It Works

- MFMA pipelines dominate total runtime
- AccVGPR read latency is partially hidden by compute
- Occupancy loss hurts less than memory stalls would

### Architectural Outcome

- Compute saturation masks some structural inefficiencies

### Verdict

✅ **Acceptable when compute-bound**

---

## 7. Case 5 — Training Workloads

### Condition
- Training rather than latency-critical inference
- Large batch sizes
- Persistent reuse of Q/K/V and weights

### Why It Works

- Kernel launch overhead is amortized
- Reduced global memory traffic is valuable
- Bandwidth savings outweigh occupancy loss

### Architectural Outcome

- Throughput remains high
- Latency hiding is less critical than bandwidth efficiency

### Verdict

✅ **Often favorable for training**

---

## 8. Case 6 — When Alternatives Are Worse

### Condition
- Two-phase softmax significantly increases global memory traffic
- Split-K introduces excessive synchronization or bandwidth overhead

### Why It Works

- FlashAttention avoids:
  - Extra kernel launches
  - Large intermediate tensors
- Occupancy loss may be preferable to bandwidth saturation

### Architectural Outcome

- Net performance still favors FlashAttention

### Verdict

⚠️ **FlashAttention may be the least-bad option**

---

## 9. When FlashAttention Is *Not* the Right Choice

FlashAttention becomes a poor fit when **multiple stressors coincide**:

- Long sequence lengths
- Large tile sizes
- Large FP32 accumulators
- Online softmax with long-lived state
- Occupancy drops to 1 wave per SIMD
- MFMA pipelines stall on accumulator reads

These conditions trigger **architectural failure modes**, not tuning issues.

---

## 10. Summary Decision Matrix

| Scenario | FlashAttention Viability |
|--------|--------------------------|
| Short sequences | ✅ Good |
| Small tiles | ✅ Good |
| Small head dimensions | ✅ Good |
| Compute-bound kernels | ✅ Good |
| Training workloads | ✅ Good |
| Memory-bandwidth-limited | ⚠️ Depends |
| Large seq × large tiles | ❌ Poor |
| Occupancy < 2 waves/SIMD | ❌ Poor |
| Latency-critical inference | ❌ Poor |

---

## 11. Key Insight

FlashAttention is **not fundamentally incompatible** with wavefront-64 GPUs.

It is **conditionally correct**.

Problems arise only when **multiple contract violations compound simultaneously**.

---

## 12. Engineering Takeaway

The correct conclusion is not:

> “FlashAttention should be replaced on AMD.”

The correct conclusion is:

> “FlashAttention should be **selectively deployed** on AMD, based on architectural regime.”

This distinction separates **engineering judgment** from blanket optimization folklore.

---

## Status

This document defines **where FlashAttention remains effective** under wavefront-64 execution.

Together with the contract and violation analyses, it completes a **balanced,
architecture-first evaluation** of attention kernel design on AMD GPUs.
