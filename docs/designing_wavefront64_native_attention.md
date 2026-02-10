# Designing a Wavefront-64 Native Attention Kernel

This document proposes a **ground-up attention kernel design** tailored specifically for
**AMD wavefront-64 GPUs**.

Rather than adapting FlashAttention, the goal is to define an attention architecture that
**respects wavefront-64 execution physics by construction**.

This is a **design document**, not an implementation guide.

---

## 1. Design Motivation

FlashAttention optimizes for:

- Warp-32 execution
- Unified register files
- Gradual occupancy degradation

AMD wavefront-64 GPUs exhibit different constraints:

- Wide execution groups (64 lanes)
- Physically split register files (ArchVGPR / AccVGPR)
- Sharp occupancy cliffs
- Explicit MFMA ↔ ALU domain crossings

Attempting to “patch” FlashAttention on AMD leads to diminishing returns.

**A native design is required.**

---

## 2. Design Principles (Non-Negotiable)

A wavefront-64-native attention kernel must satisfy:

1. **Occupancy-first design**
   - ≥2 wavefronts per SIMD is mandatory
2. **Bounded accumulator lifetime**
   - No accumulator lives across the full KV range
3. **Explicit domain management**
   - MFMA and ALU phases are scheduled, not interleaved implicitly
4. **Reduction minimization**
   - Reductions are amortized or deferred
5. **Predictable performance scaling**
   - No hidden occupancy cliffs

These principles override algorithmic elegance.

---

## 3. Architectural Decomposition

### 3.1 Phase Separation

Instead of a single monolithic kernel, the design separates attention into **explicit phases**:

1. Score accumulation
2. Normalization
3. Value accumulation

Each phase has **independent register and occupancy budgets**.

---

## 4. Phase 1 — Tiled Score Accumulation (MFMA-Dominant)

### Objective
Compute partial attention scores without committing to full normalization.

### Key Characteristics

- MFMA-heavy
- Accumulators live **only within a tile window**
- Accumulators reside in AccVGPRs
- No softmax in this phase

### Design Choices

- Tile sizes chosen to:
  - Fit AccVGPR budget
  - Preserve ≥2 wavefronts/SIMD
- Partial scores written to LDS or global memory

### Outcome

- Short accumulator lifetime
- Predictable register usage
- High MFMA utilization

---

## 5. Phase 2 — Blockwise Normalization (ALU-Dominant)

### Objective
Normalize partial scores using blockwise or hierarchical softmax.

### Key Characteristics

- ALU-heavy
- Uses ArchVGPRs exclusively
- No MFMA instructions

### Design Choices

- Two-pass or tree-based softmax
- Reduction depth explicitly controlled
- Domain crossings occur **only at phase boundaries**

### Outcome

- No AccVGPR pressure
- Reduced register live ranges
- Stable reduction cost on wavefront-64

---

## 6. Phase 3 — Value Accumulation (MFMA-Dominant)

### Objective
Apply normalized weights to values.

### Key Characteristics

- MFMA-heavy
- Accumulators reset per block
- No online rescaling

### Design Choices

- Independent accumulation per block
- Partial outputs merged hierarchically
- Accumulators flushed early

### Outcome

- Bounded accumulator residency
- High MFMA efficiency
- Occupancy preserved

---

## 7. Register Budget Strategy

### 7.1 AccVGPR Budget

- AccVGPRs are reserved **only** during MFMA phases
- No softmax state stored in AccVGPRs
- Accumulators are scoped to tile windows

### 7.2 ArchVGPR Budget

- Softmax state confined to normalization phase
- No long-lived mixed-domain state
- Pointer arithmetic minimized in MFMA phases

---

## 8. Wavefront-64 Mapping Strategy

### Lane Utilization

- Each wavefront handles:
  - Fewer rows
  - More independent blocks
- Avoids “fat” per-lane accumulators

### Reduction Strategy

- Favor hierarchical reductions
- Avoid frequent cross-lane sync
- Prefer block-level aggregation

---

## 9. Performance Characteristics (Expected)

| Dimension | Behavior |
|--------|---------|
| Occupancy | Stable ≥2 waves/SIMD |
| Register pressure | Bounded by phase |
| MFMA utilization | High and predictable |
| Scaling | Gradual, not cliff-based |
| Bandwidth usage | Higher than FlashAttention |
| Latency | Higher, but stable |

This design explicitly trades **bandwidth for occupancy and predictability**.

---

## 10. Comparison to FlashAttention

| Aspect | FlashAttention | Wavefront-64 Native |
|----|----|----|
| Accumulator lifetime | Full KV | Block-bounded |
| Softmax | Online | Blockwise |
| Register pressure | High & persistent | Phase-bounded |
| Occupancy behavior | Cliff-prone | Stable |
| Complexity | Low | Higher |
| Portability | High | AMD-specific |

---

## 11. When This Design Should Be Used

This kernel is preferred when:

- Sequence lengths are large
- Head dimensions are large
- Inference latency must be predictable
- Occupancy collapses under FlashAttention

FlashAttention remains valid in smaller regimes.

---

## 12. Engineering Tradeoffs

This design:

- Sacrifices elegance for control
- Accepts higher memory traffic
- Increases kernel complexity
- Requires architecture-specific tuning

These are **deliberate tradeoffs**, not drawbacks.

---

## 13. Key Insight

Wavefront-64 GPUs reward **explicit structure**.

Any attention kernel that relies on implicit behavior:
- Implicit reductions
- Implicit rescaling
- Implicit scheduling

will underperform.

A wavefront-64-native kernel must be **architected, not adapted**.

---

## 14. Closing Statement

This design does not attempt to “beat FlashAttention everywhere.”

It exists to provide:

- Predictable performance
- Architectural alignment
- A stable alternative when FlashAttention fails

That is the role of a **native kernel**.

---

## Status

This document proposes a **new attention architecture**.

Future work may include:

- Prototype kernels
- Cost-model-driven kernel selection
- Integration with runtime heuristics
