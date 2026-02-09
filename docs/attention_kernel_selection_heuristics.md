# Attention Kernel Selection Heuristics for Wavefront-64 GPUs

This document defines **practical, architecture-driven heuristics** for selecting
the appropriate attention kernel on **AMD wavefront-64 GPUs**.

It answers a critical systems question:

> Given a workload and hardware constraints, **which attention kernel should run?**

This is a **policy document**, not an optimization guide.

---

## 1. Why Kernel Selection Matters More Than Kernel Design

On wavefront-64 GPUs, no single attention kernel is optimal across all regimes.

Key reasons:

- Occupancy cliffs are sharp, not gradual
- Register pressure interacts non-linearly with wavefront width
- MFMA scheduling is sensitive to accumulator lifetime
- Bandwidth vs occupancy tradeoffs dominate performance

As a result:

> Choosing the *wrong* kernel can be worse than using a suboptimal implementation of the *right* one.

---

## 2. Kernel Design Axes That Matter on AMD

Kernel selection must consider **architectural axes**, not just algorithmic ones.

### 2.1 Primary Axes

| Axis | Why It Matters |
|----|----------------|
| Sequence length | Determines accumulator lifetime |
| Head dimension | Determines AccVGPR footprint |
| Tile size | Determines register pressure & occupancy |
| Compute vs bandwidth bound | Determines sensitivity to occupancy |
| Latency vs throughput | Changes tolerance for extra passes |

---

## 3. Candidate Kernel Families

This document considers the following kernel families:

1. **FlashAttention (Online Softmax, Single-Pass)**
2. **FlashAttention (Small-Tile Variant)**
3. **Split-K Attention**
4. **Two-Phase Softmax Attention**
5. **Conventional Fused Attention (Non-Flash)**

---

## 4. High-Level Selection Heuristics

### Heuristic 1 — Sequence Length First

| Sequence Length | Recommended Kernel |
|----------------|--------------------|
| ≤256 | FlashAttention |
| 256–512 | FlashAttention (small tiles) |
| 512–2K | Split-K or small-tile Flash |
| >2K | Two-phase softmax or alternative |

**Rationale:**  
Accumulator and softmax state lifetime scale linearly with sequence length.
Beyond a threshold, register pressure dominates.

---

### Heuristic 2 — Occupancy Is a Hard Constraint

If estimated occupancy < **2 waves per SIMD**:

- FlashAttention **must not** be used
- Switch to:
  - Smaller tiles
  - Split-K
  - Two-phase softmax

Occupancy loss on wavefront-64 produces **hard performance cliffs**, not gradual slowdowns.

---

### Heuristic 3 — Head Dimension Thresholds

| Head Dim | Recommendation |
|--------|----------------|
| ≤64 | FlashAttention viable |
| 64–128 | Caution; reduce tile size |
| >128 | Avoid online softmax |

Large head dimensions inflate both:
- AccVGPR usage (accumulators)
- ArchVGPR usage (softmax state)

---

### Heuristic 4 — Compute vs Bandwidth Bound

- **Compute-bound workloads**
  - Can tolerate lower occupancy
  - FlashAttention may remain viable
- **Bandwidth-bound workloads**
  - Prefer kernels with higher occupancy
  - Avoid long-lived accumulators

---

### Heuristic 5 — Latency Sensitivity

| Workload | Recommendation |
|--------|----------------|
| Training | FlashAttention often acceptable |
| Inference (batch) | Depends on shape |
| Inference (latency-critical) | Avoid FlashAttention |

Single-pass kernels optimize bandwidth, not tail latency.

---

## 5. Decision Matrix

| Regime | Preferred Kernel |
|------|------------------|
| Short seq + small heads | FlashAttention |
| Medium seq + moderate heads | Small-tile Flash |
| Long seq | Split-K |
| Very long seq | Two-phase softmax |
| Latency-critical inference | Non-Flash fused |
| Occupancy < 2 waves/SIMD | Never FlashAttention |

---

## 6. Compiler and Runtime Implications

These heuristics should ideally be enforced by:

- Kernel dispatch logic
- Autotuning constraints
- Shape-based guards
- Runtime kernel selection

Blindly dispatching FlashAttention based solely on sequence length
is insufficient on wavefront-64 GPUs.

---

## 7. Key Insight

On AMD GPUs:

> **Kernel selection is a first-class optimization.**

Performance is determined as much by **what you do not run**
as by how well a kernel is optimized.

---

## 8. Relationship to Other Documents

This document builds on:

- `attention_kernel_contract_for_wavefront64.md`
- `where_flashattention_violates_wavefront64_contract.md`
- `when_flashattention_is_still_effective_on_wavefront64.md`

It provides the **decision logic** that connects architectural analysis
to real systems.

---

## Status

This document defines **selection policy**, not kernel implementations.

Next steps naturally include:

- Designing a wavefront-64-native attention kernel
- Identifying which FlashAttention violations are worth fixing
- Encoding these heuristics into compiler or runtime logic
