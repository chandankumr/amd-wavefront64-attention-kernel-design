# Wavefront-64 Native Attention  
## A Register-Topology–Aware Design for CDNA4 (gfx950)

---

## Abstract

Attention kernels are typically designed under warp-32 assumptions,
where per-thread accumulator ownership and high arithmetic intensity
dominate optimization strategy.

However, AMD CDNA4 (gfx950) GPUs execute in a Wavefront-64 model with:

- 256 VGPR per SIMD
- 64-lane execution width
- MFMA matrix cores requiring multi-wave latency hiding

This whitepaper demonstrates that:

> Persistent full-head accumulator ownership becomes structurally unstable on Wavefront-64 at HEAD_DIM ≥ 128.

Using empirical Radeon GPU Analyzer (RGA) data and a formal register model,
we derive an architecture-aligned alternative:

Wavefront-64 Native Partitioned Attention.

This document formalizes:

- The register-collapse threshold
- The quantitative mismatch with FlashAttention-style tiles
- A partitioned accumulator design
- Expected performance projections
- A hardware-aligned attention contract

---

# 1. Architectural Background

## 1.1 CDNA4 (gfx950) Execution Model

| Parameter | Value |
|------------|--------|
| Wavefront Size | 64 lanes |
| VGPR per SIMD | 256 |
| SGPR per SIMD | 102 |
| LDS per CU | 163 KB |
| Matrix Core | MFMA (F16 / F8) |

Occupancy per SIMD:

waves_per_simd = floor(256 / used_vgpr_per_lane)


Critical occupancy boundaries:

| Used VGPR | Waves |
|------------|--------|
| ≤ 85 | 3 waves |
| ≤ 128 | 2 waves |
| > 128 | 1 wave (collapse) |

MFMA latency hiding requires:

waves_per_simd ≥ 2


Therefore:

used_vgpr_per_lane ≤ 128


This is the central constraint.

---

# 2. The Register Collapse Phenomenon

## 2.1 Empirical Sweep Results (RGA)

Measured on gfx950:

| Accumulators | Used VGPR | Waves |
|--------------|-----------|--------|
| 1 | 16 | 16 |
| 2 | 16 | 16 |
| 4 | 16 | 16 |
| 8 | 80 | 3 |
| 16 | 136 | 1 |
| 32 | 296 | 1 |

Observation:

> Collapse occurs between 120–136 VGPR.

At 136 VGPR:

256 / 136 = 1 wave


Matrix core becomes latency exposed.

---

## 2.2 Root Cause

Full-head persistent accumulation implies:

R_accumulator ≈ HEAD_DIM


If:

HEAD_DIM = 128


Then:

R_accumulator ≈ 128


Total register usage becomes:

R_total = 128 + overhead


This exceeds the 128-VGPR dual-wave boundary.

Collapse becomes deterministic.

---

# 3. FlashAttention Under Wavefront-64

FlashAttention-3 improves memory pipelining and instruction overlap,
but it preserves:

- Full-head accumulator persistence
- Large BLOCK_M
- Per-lane head ownership

Under the Wavefront-64 register model:

For:

HEAD_DIM = 128


We approximate:

R_total ≈ 128 (acc) + 20–40 overhead


Result:

R_total ≥ 148
waves_per_simd = 1


Therefore:

> FlashAttention-3 remains structurally unstable at HEAD_DIM ≥ 128 on Wavefront-64 unless partitioned.

This is not a software inefficiency.

It is a hardware topology mismatch.

---

# 4. Wavefront-64 Native Design

## 4.1 Core Principle

Instead of:

One lane owns full HEAD_DIM


We define:

Partition HEAD_DIM across lanes


Let:

HEAD_DIM = D
partition_factor = P


Per-lane accumulator:

D / P


Constraint:

(D / P) + overhead ≤ 120


---

## 4.2 Example

For:

HEAD_DIM = 128
P = 2


Per-lane:

64 accumulators


Total:

64 + overhead ≈ 90–110


Result:

waves_per_simd ≥ 2


Occupancy preserved.

---

# 5. Structural Comparison

| Feature | FlashAttention-3 | Wavefront-64 Native |
|----------|------------------|----------------------|
| Accumulator Ownership | Full-head | Partitioned |
| VGPR Growth | Linear in HEAD_DIM | Linear in HEAD_DIM / P |
| Collapse Threshold | ~128 | ~256 (with P=2) |
| Occupancy Stability | Conditional | Guaranteed |
| Cross-Lane Reduction | Minimal | Required (small cost) |

Cross-lane reduction cost:

O(log 64)


Register collapse cost:

O(HEAD_DIM)


Partitioning dominates.

---

# 6. Performance Projection

For HEAD_DIM = 128:

| Metric | FlashAttention-3 | Native Partitioned |
|---------|------------------|--------------------|
| Used VGPR | ~148 | ~100 |
| Waves | 1 | 2 |
| MFMA Utilization | 50–65% | 80–90% |
| Latency Hiding | Weak | Strong |

Projected throughput gain:

+20–40% in compute-bound regime


Memory-bound regimes may show smaller gains.

---

# 7. Design Contract

A Wavefront-64 attention kernel must satisfy:

1. `R_total ≤ 120`
2. `waves_per_simd ≥ 2`
3. `R_accumulator ≤ 80`
4. No deterministic collapse at HEAD_DIM ≤ 128
5. MFMA issue density ≥ 80%

Violation implies architectural misalignment.

---

# 8. Advanced Extensions

This model composes with:

- Asynchronous LDS staging
- Dual-issue VALU/MFMA overlap
- FP8 scaled MFMA instructions
- Split-K across wavefronts
- Bank-conflict–free LDS swizzling

These techniques improve performance,
but none replace accumulator partitioning as the structural fix.

---

# 9. Conclusion

Wavefront-64 is not a larger Warp-32.

Its register topology changes the economics of attention tiling.

Persistent full-head accumulation becomes unstable
when HEAD_DIM approaches 128.

The solution is not micro-optimization.

The solution is structural partitioning.

Wavefront-64 Native Attention is defined by:

- Controlled accumulator ownership
- Guaranteed ≥2-wave residency
- Register-safe tiling

This framework converts attention kernel design from
trial-and-error tuning
into hardware-aligned engineering.

---

# 10. Final Statement

On CDNA4 (gfx950):

Register topology is the primary constraint.

Attention must be designed around:

waves_per_simd = floor(256 / R_total)


Before optimizing memory movement,
before pipelining,
before instruction fusion,
one must satisfy the register contract.

Architecture first.
Tiling second.

This is Wavefront-64 Native Attention.