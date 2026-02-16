# Wavefront-64 Attention Design Contract  
## Architectural Rules for CDNA4 (gfx950) Attention Kernels

---

## 1. Purpose

This document defines a **formal architectural contract** for designing
high-performance attention kernels on:

- CDNA4 (gfx950)
- Wavefront-64 execution model
- 256 VGPR per SIMD constraint

The goal is to move attention kernel design from:

"Empirical tuning"

to

"Architecturally grounded engineering."

---

## 2. Hardware Contract (CDNA4)

### 2.1 Execution Model

| Parameter | Value |
|------------|--------|
| Wavefront size | 64 lanes |
| VGPR per SIMD | 256 |
| SGPR per SIMD | 102 |
| LDS per CU | 163 KB |
| MFMA unit | Dedicated matrix core |

---

### 2.2 Occupancy Equation

Occupancy per SIMD:

waves_per_simd = floor(256 / used_vgpr_per_lane)


Critical thresholds:

| Used VGPR | Waves |
|------------|--------|
| ≤ 85 | 3 waves |
| ≤ 128 | 2 waves |
| > 128 | 1 wave (collapse) |

Full MFMA efficiency requires:

waves_per_simd ≥ 2


Therefore:

used_vgpr_per_lane ≤ 128


This is a hard architectural constraint.

---

## 3. Attention Kernel Resource Model

Per-lane register consumption is approximately:

R_total =
R_accumulator +
R_Q_fragment +
R_K_fragment +
R_softmax_state +
R_temporaries


Where:

R_accumulator ≈ HEAD_DIM / partition_factor


This term dominates.

---

## 4. Contract Rule #1 — No Full-Head Ownership Per Lane

Forbidden pattern:

float acc[HEAD_DIM];


If:

HEAD_DIM ≥ 128


This causes:

R_accumulator ≥ 128
→ occupancy collapse


Therefore:

> A single lane must not own the full HEAD_DIM when HEAD_DIM ≥ 96–128.

---

## 5. Contract Rule #2 — Maintain ≥ 2 Waves Per SIMD

Design target:

R_total ≤ 120 VGPR


This preserves:

waves_per_simd ≥ 2


Rationale:

- Hides MFMA latency
- Allows scheduler interleaving
- Prevents matrix-core starvation

Single-wave execution is considered:

> Structurally unstable for large-head attention.

---

## 6. Contract Rule #3 — Partition Accumulator Across Lanes

Let:

HEAD_DIM = D
partition_factor = P


Then:

Per-lane accumulators = D / P


Constraint:

(D / P) + overhead ≤ 120


Example:

| HEAD_DIM | Partition | Per-lane acc | Safe? |
|------------|-----------|---------------|--------|
| 64 | 1 | 64 | Yes |
| 128 | 1 | 128 | No |
| 128 | 2 | 64 | Yes |
| 256 | 2 | 128 | No |
| 256 | 4 | 64 | Yes |

Partitioning is the primary register control mechanism.

---

## 7. Contract Rule #4 — Prefer Structural Over Temporal Reduction

Bad pattern:

- Accumulate full head
- Reduce later

Preferred pattern:

- Structurally split ownership
- Cross-lane reduction only once

Cross-lane reduction cost is:

O(log wave_size)


Register pressure cost of not splitting is:

O(HEAD_DIM)


Structural savings dominate.

---

## 8. Contract Rule #5 — MFMA Must Remain Compute-Bound

A valid kernel must satisfy:

- ≥ 2 waves
- No AGPR spilling
- No excessive VGPR spilling
- MFMA issue density ≥ 80%

If occupancy collapse occurs:

- MFMA utilization drops
- Latency becomes exposed
- Arithmetic intensity no longer matters

---

## 9. Contract Rule #6 — LDS Usage Must Support Partitioning

When partitioning HEAD_DIM:

- LDS layout must allow coalesced loads
- Cross-lane data sharing must avoid bank conflicts
- Transpose operations must not reintroduce VGPR inflation

LDS is a coordination tool,
not a register spill buffer.

---

## 10. Contract Rule #7 — FP8 Is an Optional Multiplier, Not a Structural Fix

Using FP8:

- Reduces accumulator footprint
- Increases MFMA throughput

But:

If full-head ownership remains,
collapse can still occur at larger HEAD_DIM.

Precision reduction does not replace structural partitioning.

---

## 11. FlashAttention Compatibility Clause

FlashAttention-style persistent tiles are valid when:

HEAD_DIM ≤ 64


Or:

Memory-bound regime dominates


This contract does not invalidate FlashAttention.

It defines when it becomes structurally misaligned with Wavefront-64 economics.

---

## 12. Formal Design Objective

A Wavefront-64-native attention kernel must satisfy:

1. `R_total ≤ 120`
2. `waves_per_simd ≥ 2`
3. `R_accumulator ≤ 80`
4. MFMA utilization ≥ 80%
5. No deterministic occupancy collapse at HEAD_DIM ≤ 128

If any of these fail,
the design violates the Wavefront-64 contract.

---

## 13. Architectural Philosophy

Warp-32 GPUs optimize:

- Per-thread ownership
- High arithmetic intensity
- Deep pipelining

Wavefront-64 GPUs optimize:

- Balanced per-lane register footprint
- Multi-wave latency hiding
- Structural partitioning

Therefore:

Attention tiling is not universal.

It is architecture-dependent.

---

## 14. Final Statement

On CDNA4 (gfx950):

Register topology is the primary constraint.

Attention kernel design must be treated as:

> A register-allocation architecture problem  
before it is treated as a tiling or algorithm problem.

Wavefront-64-native attention is defined by:

- Controlled accumulator ownership
- Guaranteed dual-wave residency
- Structural register safety

This is the Wavefront-64 Attention Design Contract.

---