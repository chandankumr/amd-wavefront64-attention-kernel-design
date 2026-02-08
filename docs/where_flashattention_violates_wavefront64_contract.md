# Where FlashAttention Violates the Wavefront-64 Kernel Contract

This document evaluates **FlashAttention-style kernels** against the architectural
constraints defined in:

> `attention_kernel_contract_for_wavefront64.md`

The purpose is **not to criticize the algorithm**, but to identify **structural mismatches**
between FlashAttention’s design assumptions and **AMD wavefront-64 execution**.

These violations are **predictable, repeatable, and architectural**.

---

## 1. Contract Reference

This analysis assumes the following **non-negotiable requirements** from the contract:

- ≥2 resident wavefronts per SIMD required for latency hiding
- Explicit AccVGPR ↔ ArchVGPR movement
- Bounded accumulator lifetime
- Reduction cost awareness
- Occupancy-first design

Any violation below is evaluated against these requirements.

---

## 2. Violation #1 — Unbounded Accumulator Lifetime

### Contract Clause
> *Accumulator lifetime must be bounded to preserve occupancy.*

### FlashAttention Behavior

FlashAttention maintains a **single FP32 output accumulator** (`acc_o`) that:

- Is initialized once
- Persists across **all KV blocks**
- Cannot be written back early due to online softmax rescaling
- Is updated incrementally for the full sequence

### Architectural Consequence on Wavefront-64

- Accumulators remain live in **AccVGPRs** for the entire kernel
- Register usage scales ×64 lanes per wavefront
- Occupancy drops sharply once VGPR/AGPR thresholds are crossed

### Result

**Violation:** Long-lived accumulators directly violate the bounded residency contract.

This is not a tuning issue — it is a structural property of the algorithm.

---

## 3. Violation #2 — Online Softmax Dependency Chain

### Contract Clause
> *Softmax state lifetime must be explicitly accounted for.*

### FlashAttention Behavior

Online softmax requires:

- Persistent running maximum (`m`)
- Persistent normalization denominator (`l`)
- Rescaling of the output accumulator whenever a new max is encountered

These values:

- Remain live across all KV tiles
- Prevent early accumulator writeback
- Extend ArchVGPR live ranges

### Architectural Consequence on Wavefront-64

- Softmax state compounds register pressure
- AccVGPR → ArchVGPR reads occur repeatedly
- MFMA scheduling becomes interleaved with accumulator reads

### Result

**Violation:** Online softmax enforces long-lived state incompatible with wavefront-64 occupancy requirements.

---

## 4. Violation #3 — AccVGPR ↔ ArchVGPR Domain Crossing Frequency

### Contract Clause
> *Domain crossing is explicit and must be amortized.*

### FlashAttention Behavior

Each KV block requires:

1. MFMA (Q × Kᵀ) → AccVGPRs
2. `v_accvgpr_read` to ArchVGPRs
3. Softmax math (exp, sum, normalize)
4. Rescaled accumulation

This crossing occurs **inside the main loop**, not amortized across phases.

### Architectural Consequence on Wavefront-64

- `v_accvgpr_read` has fixed latency
- MFMA units may idle during accumulator reads
- Low occupancy exacerbates pipeline bubbles

### Result

**Violation:** FlashAttention assumes domain crossings can be hidden by warp-level scheduling, which does not hold on wavefront-64.

---

## 5. Violation #4 — Occupancy as a Secondary Objective

### Contract Clause
> *Occupancy must be a first-class design goal.*

### FlashAttention Behavior

FlashAttention optimizes for:

- Minimal memory traffic
- Maximal data reuse
- Large tile sizes

These choices implicitly assume:

- Unified register files
- Gradual occupancy degradation
- Warp-32 scheduling flexibility

### Architectural Consequence on Wavefront-64

- Occupancy loss occurs as a **step function**
- Dropping from 2 waves → 1 wave exposes latency immediately
- Performance cliffs appear instead of smooth degradation

### Result

**Violation:** FlashAttention prioritizes bandwidth over occupancy, which is misaligned with wavefront-64 execution.

---

## 6. Violation #5 — Reduction Cost Underestimation

### Contract Clause
> *Reduction cost must be evaluated for wavefront-64.*

### FlashAttention Behavior

Reductions are frequent in:

- Softmax max computation
- Sum of exponentials
- Masking and normalization

Design assumptions are inherited from warp-32 systems.

### Architectural Consequence on Wavefront-64

- More reduction steps
- Higher LDS pressure
- Greater sensitivity to register pressure

### Result

**Violation:** Reduction-heavy phases become latency-sensitive under wavefront-64 when combined with high register usage.

---

## 7. Summary of Violations

| Contract Requirement | FlashAttention Status |
|---------------------|----------------------|
| Bounded accumulator lifetime | ❌ Violated |
| Softmax state containment | ❌ Violated |
| Domain crossing amortization | ❌ Violated |
| Occupancy-first design | ❌ Violated |
| Wavefront-aware reductions | ❌ Violated |

None of these violations are accidental.

---

## 8. Key Insight

FlashAttention’s design is **internally consistent** and **algorithmically optimal**
under its original assumptions.

However, those assumptions implicitly target:

- Warp-32 execution
- Unified register files
- Gradual occupancy loss

Wavefront-64 GPUs violate all three.

---

## 9. What This Does *Not* Imply

This analysis does **not** imply:

- FlashAttention is “bad”
- AMD hardware is “inferior”
- Compilers are “immature”

It implies that **FlashAttention is not architecture-neutral**.

---

## 10. Consequence

On wavefront-64 GPUs, FlashAttention should be treated as:

- A **reference algorithm**
- Not a drop-in kernel design

Architecturally compliant alternatives must:

- Shorten accumulator lifetimes
- Decouple softmax phases
- Accept bandwidth to preserve occupancy

---

## Status

This document identifies **where** FlashAttention violates the wavefront-64 contract.

Subsequent documents will explore:

- Which violations are tolerable
- Which must be redesigned
- How alternative attention formulations satisfy the contract


## Appendix A — ISA Evidence (MFMA + AccVGPR Reads)

![ScreenShot](../experiments/rga/mfma_minimal/rga_mfma_isa.png)

This ISA snippet illustrates:
- MFMA accumulation into AccVGPRs
- Explicit `v_accvgpr_read`
- Interleaving of accumulator reads with ALU ops
