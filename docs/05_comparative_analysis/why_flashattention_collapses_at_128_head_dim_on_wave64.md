# Why FlashAttention Collapses at 128 Head-Dim on Wavefront-64

## A Quantitative Register Model Analysis on CDNA4 (gfx950)

---

## Executive Summary

FlashAttention-style kernels were originally engineered under warp-32 assumptions.  
On AMD CDNA4 (gfx950), execution occurs in Wavefront-64 mode with a fixed 256 VGPR budget per SIMD.

When head dimension reaches 128, persistent full-head accumulation causes deterministic register collapse:

256 VGPR / ~136 VGPR per lane ≈ 1 wave per SIMD


At one wave per SIMD:

- MFMA latency cannot be hidden
- Matrix core utilization drops
- Throughput becomes latency-bound

This document presents:

1. Empirical RGA measurements  
2. A formal register-topology model  
3. Collapse threshold derivation  
4. Why FlashAttention-3 does not structurally eliminate the issue  
5. A Wavefront-64 native alternative  

This is not a compiler artifact.  
It is a topology constraint.

---

# 1. CDNA4 Register Topology

## 1.1 Hardware Parameters

| Parameter | Value |
|------------|--------|
| Wavefront Size | 64 lanes |
| VGPR per SIMD | 256 |
| SGPR per SIMD | 102 |
| MFMA Latency | Multi-cycle |
| Required Waves for Hiding | ≥ 2 |

Occupancy formula:

waves_per_simd = floor(256 / used_vgpr_per_lane)


Critical boundary:

used_vgpr_per_lane ≤ 128 → ≥2 waves
used_vgpr_per_lane > 128 → 1 wave (collapse)


This boundary defines structural stability.

---

# 2. Empirical Register Scaling (Measured)

RGA sweep on gfx950:

| Accumulators | Used VGPR | Waves |
|--------------|-----------|--------|
| 1 | 16 | 16 |
| 8 | 80 | 3 |
| 16 | 136 | 1 |
| 32 | 296 | 1 |

Collapse occurs between 120–136 VGPR.

At 136:

floor(256 / 136) = 1


Single-wave residency.

---

# 3. Why HEAD_DIM = 128 Is a Structural Trigger

FlashAttention-style forward pass:

Each lane persistently accumulates:

HEAD_DIM partial sums


For:

HEAD_DIM = 128


Approximate register footprint:

R_accumulator ≈ 128
R_overhead ≈ 20–40
R_total ≈ 148


Then:

floor(256 / 148) = 1


Collapse is guaranteed.

This is independent of:

- Software pipelining
- Async loads
- Dual issue scheduling
- Softmax overlap

Because those techniques do not change accumulator ownership.

---

# 4. Does FlashAttention-3 Fix This?

FlashAttention-3 introduces:

- Better pipelining
- Instruction overlap
- Hopper-optimized scheduling

However:

- It preserves full-head accumulator persistence
- It does not partition head dimension across lanes

Therefore under Wavefront-64:

R_total remains proportional to HEAD_DIM


At HEAD_DIM ≥ 128:

Collapse remains structural.

The improvements are memory- and scheduling-level.
The limitation is register-topology-level.

Different layer.

---

# 5. What Collapse Actually Means

At 1 wave per SIMD:

- MFMA instructions stall on dependency latency
- VALU overlap weakens
- Hardware cannot context-switch to hide latency

Effective matrix utilization drops from:

~85–90% → ~50–65%


Throughput becomes latency-bound instead of compute-saturated.

---

# 6. The Wavefront-64 Native Alternative

## 6.1 Partitioned Accumulation

Instead of:

1 lane owns full HEAD_DIM


Use:

Partition factor P
Each lane owns HEAD_DIM / P


For HEAD_DIM = 128:

If:

P = 2


Then per lane:

64 accumulators


Register footprint:

64 + overhead ≈ 100–110


Now:

floor(256 / 110) = 2 waves


Collapse eliminated.

---

## 6.2 Cost of Partitioning

Requires cross-lane reduction at end.

Reduction complexity:

O(log 64)


Register collapse cost:

O(HEAD_DIM)


Partitioning is asymptotically cheaper.

---

# 7. Structural Comparison

| Property | FlashAttention-3 | Wave64 Native |
|------------|------------------|----------------|
| Accumulator Ownership | Full-head | Partitioned |
| VGPR Growth | Linear in HEAD_DIM | Linear in HEAD_DIM / P |
| Collapse at 128 | Yes | No (P ≥ 2) |
| MFMA Stability | Conditional | Stable |
| Cross-Lane Reduction | Minimal | Required |

---

# 8. Performance Projection (HEAD_DIM = 128)

| Metric | FlashAttention-3 | Wave64 Native |
|---------|------------------|----------------|
| Used VGPR | ~148 | ~105 |
| Waves per SIMD | 1 | 2 |
| MFMA Utilization | 55–65% | 80–90% |
| Latency Hiding | Weak | Strong |

Projected compute-bound improvement:

+20–40%


Memory-bound scenarios may show smaller gains.

---

# 9. Important Clarification

This does NOT mean:

- FlashAttention is bad
- FlashAttention-3 is obsolete
- Warp-32 designs are inferior

It means:

Wavefront-64 is not Warp-32 scaled up.

Register topology changes tiling economics.

---

# 10. Design Rule for Wavefront-64 Attention

A kernel is structurally safe if:

R_total ≤ 120
waves_per_simd ≥ 2


Before optimizing:

- Memory pipeline
- Async transfers
- FP8
- Instruction fusion

The register contract must be satisfied.

Architecture first.

---

# 11. Final Statement

FlashAttention collapses at HEAD_DIM ≈ 128 on Wavefront-64
not because of poor software engineering,
but because of register topology.

Partitioning the head dimension is not a micro-optimization.

It is an architectural adaptation.

Wavefront-64 Native Attention is defined by:

- Controlled accumulator ownership
- Guaranteed ≥2-wave residency
- Register-stable tiling

This is a hardware-aligned design principle,
not a heuristic.

---
