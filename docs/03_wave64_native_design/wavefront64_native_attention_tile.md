# Wavefront-64 Native Attention Tile Design (CDNA4)

---

## 1. Objective

This document proposes a **wavefront-64-native attention tile** designed specifically for:

- Architecture: CDNA4 (gfx950)
- Wavefront size: 64
- VGPR per SIMD: 256
- Allocation granularity: 8

The goal is to eliminate the structural occupancy collapse observed when running
FlashAttention-style persistent accumulators on wave-64 hardware.

---

## 2. The Core Problem

FlashAttention (v1–v3) uses a **persistent row accumulator**:

Each lane holds:

Head_Dim FP32 accumulators


For:

Head_Dim = 128


Per-lane accumulator cost:

128 VGPR


Under CDNA4 economics:

Total_VGPR ≈ 128 + overhead ≈ 152


Occupancy:

floor(256 / 152) = 1 wave


This is structural occupancy collapse.

---

## 3. Design Principle: Split the Head Dimension Across Lanes

Instead of:

1 lane → 1 full row (Head_Dim elements)


We use:

2 lanes → 1 row


Each lane stores:

Head_Dim / 2


For Head_Dim = 128:

Per-lane accumulator = 64 FP32 values


Accumulator cost per lane:

64 VGPR


---

## 4. Register Model Under Split-Head Design

Total per-lane usage:

Accumulator ≈ 64
Q fragment ≈ 8–12
K fragment ≈ 8–12
Softmax temporaries ≈ 8
Loop bookkeeping ≈ 4
Estimated total ≈ 96–104 VGPR


Occupancy:

floor(256 / 104) = 2 waves


This restores dual-wave residency.

---

## 5. Lane Mapping Strategy

Let:

BLOCK_M = 64
Head_Dim = 128


Wave mapping:

| Lane ID | Row ID | Head Slice |
|----------|--------|------------|
| 0        | 0      | 0–63       |
| 1        | 0      | 64–127     |
| 2        | 1      | 0–63       |
| 3        | 1      | 64–127     |
| ...      | ...    | ...        |
| 62       | 31     | 0–63       |
| 63       | 31     | 64–127     |

Effective rows per wave:

BLOCK_M_effective = 32


But compute density remains high because:

- MFMA operations remain fully utilized
- All 64 lanes participate in matrix math

---

## 6. Cross-Lane Reduction Cost

After computing partial accumulators, we must combine:

Lane_even + Lane_odd


Reduction method:

- ds_bpermute
- or wave-level shuffle
- or LDS reduction (if multi-wave)

Cost:

- O(log2(2)) = 1 shuffle
- Negligible compared to MFMA cost

This overhead is tiny relative to occupancy gain.

---

## 7. MFMA Scheduling Compatibility

This design preserves:

- Full MFMA width utilization
- Stationary-A or Stationary-B scheduling
- Accumulator residency within AGPR domain

No structural conflict with matrix core pipelines.

---

## 8. Memory Behavior

Advantages:

- Smaller per-lane accumulator footprint
- Reduced VGPR live range
- Better compiler scheduling freedom
- Improved latency hiding due to dual-wave occupancy

This reduces memory stall amplification under wave-64.

---

## 9. Generalization Rule

For arbitrary Head_Dim:

Let:

Partition_Factor = ceil(Head_Dim / Safe_Accumulator_Limit)


Where Safe_Accumulator_Limit on CDNA4 ≈ 64–80 FP32.

Per-lane accumulator:

Head_Dim / Partition_Factor


This guarantees:

Total_VGPR ≤ 120


Maintaining ≥2 wave occupancy.

---

## 10. Comparison: FlashAttention vs Wavefront64-Native

| Feature | FlashAttention | Wave64-Native |
|----------|---------------|---------------|
| Accumulator | Persistent full row | Partitioned row |
| Per-lane VGPR | ~128 | ~64 |
| Occupancy | 1 wave | 2 waves |
| Cross-lane cost | None | Minimal |
| CDNA4 efficiency | Reduced | Restored |

---

## 11. Architectural Insight

This design is not an optimization tweak.

It is an **architecture-aligned mapping**.

FlashAttention assumes warp-32 economics.

CDNA4 requires wave-64-aware register residency control.

The difference is structural, not stylistic.

---

## 12. Conclusion

A wavefront-64-native attention tile:

- Preserves MFMA efficiency
- Avoids occupancy collapse
- Controls accumulator lifetime
- Matches CDNA4 register economics

This is the minimal architectural modification required
to make attention kernels wave-64 optimal.

---
