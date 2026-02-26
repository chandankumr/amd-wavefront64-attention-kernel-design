# NVIDIA Warp-32 vs AMD Wavefront-64 Register Model

## 1. Purpose

This document contrasts:

- NVIDIA Warp-32 register behavior
- AMD Wavefront-64 register behavior

to explain why FlashAttention scales well on NVIDIA GPUs but encounters structural pressure on CDNA-class architectures.

---

## 2. Execution Granularity

### NVIDIA (RTX 4070 / Ada)

- Warp size: 32 threads
- Registers per SM: 256K 32-bit registers
- Allocation per thread
- Typical register limit per thread: 255

### AMD CDNA4 (gfx950)

- Wavefront size: 64 lanes
- Registers per SIMD: 256 VGPRs
- Allocation per wave
- Allocation granularity: 8 VGPR

Key difference:

NVIDIA allocates per-thread  
AMD allocates per-wave

---

## 3. FlashAttention Register Model (Warp-32)

Assume:

HEAD_DIM = 128

On NVIDIA:

Each thread owns full HEAD_DIM slice.

Registers per thread ≈ 128 accumulators + temporaries.

Total register file per SM is large enough to sustain multiple warps even at 128 registers per thread.

Occupancy degrades gradually.

---

## 4. FlashAttention Under Wavefront-64

Wave size doubles from 32 → 64.

Each wave must allocate:

HEAD_DIM accumulators per lane.

For HEAD_DIM = 128:

VGPR_per_wave ≈ 128 + temporaries  
Rounded to allocation granularity.

Given:

VGPR_total = 256 per SIMD

Waves per SIMD:

floor(256 / VGPR_per_wave)

At VGPR ≈ 136 → 1–2 waves  
At VGPR ≈ 296 → 1 wave (collapse)

This is a hard architectural limit.

---

## 5. Structural Mismatch Summary

FlashAttention assumes:

- Warp-32
- Per-thread register scaling
- Large register file per SM

CDNA assumes:

- Wave-64
- Per-wave allocation
- Fixed 256 VGPR per SIMD

Thus:

A kernel tuned for Warp-32 does not automatically scale to Wave-64.

The mismatch is architectural, not algorithmic.

---

## 6. Implication

FlashAttention-3 improvements:

- Reduce memory traffic
- Improve softmax fusion
- Improve tiling

But do not eliminate:

Wavefront-64 register pressure amplification.

Therefore, architectural re-partitioning (split-head, cross-lane reduction) becomes necessary for high HEAD_DIM values.

---

## 7. Balanced View

FlashAttention remains optimal when:

- HEAD_DIM ≤ 64
- VGPR_per_wave ≤ 120
- Waves per SIMD ≥ 2

Collapse occurs when:

HEAD_DIM ≥ 128  
Persistent accumulators dominate register file

This threshold is architecture-dependent.