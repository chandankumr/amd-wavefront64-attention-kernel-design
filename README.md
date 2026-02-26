# FlashAttention vs Wavefront-64  
## A Quantitative Register-Level Analysis on AMD CDNA4 (gfx950)

---

## 1. Executive Summary

FlashAttention is highly optimized for NVIDIA Warp-32 architectures.

This repository demonstrates, using compiler-level analysis and register accounting, that:

> FlashAttention-style persistent accumulator designs encounter structural occupancy collapse on AMD Wavefront-64 architectures at HEAD_DIM ≥ 128.

The issue is not algorithmic.  
It is architectural.

Using Radeon GPU Analyzer (RGA) output, we show:

- VGPR usage scaling with accumulator count
- Allocation granularity effects
- Wave occupancy collapse at 256 VGPR per SIMD
- A wavefront-64 native alternative design

---

## 2. Key Finding

On CDNA4 (gfx950):

Total VGPR per SIMD = 256
Waves per SIMD = floor(256 / VGPR_per_wave)


From actual RGA measurements:

| HEAD_DIM | USED_VGPR | Waves/SIMD |
|-----------|-----------|------------|
| 32        | 80        | 3          |
| 64        | 136       | 1–2        |
| 128       | 296       | 1 ← Collapse |

When VGPR usage exceeds 256:

→ Only one wave can reside on a SIMD  
→ Latency hiding collapses  
→ MFMA utilization drops  

This is a hard hardware constraint.

---

## 3. Why This Matters

FlashAttention assumes:

- Warp-32 execution
- Per-thread register scaling
- Large register file per SM

CDNA assumes:

- Wavefront-64 execution
- Per-wave register allocation
- 256 VGPR per SIMD limit
- 8-VGPR allocation granularity

Doubling warp width from 32 → 64 amplifies persistent accumulator pressure.

This creates an architectural mismatch.

---

## 4. Repository Structure

### 📄 Core Analysis

- `docs/cdna4_wavefront64_register_model.md`
- `docs/flashattention3_under_cdna4_register_model.md`
- `docs/wavefront64_native_attention_tile.md`
- `docs/wavefront64_native_vs_flashattention3_summary.md`
- `docs/why_flashattention_collapses_at_128_head_dim_on_wave64.md`

### 📊 Compiler-Verified Experiments

- `experiments/rga/mfma_occupancy_sweep/`
- `experiments/rga/mfma_vgpr_scaling/`
- `experiments/rga/mfma_spill_boundary/`

Each experiment includes:

- ISA disassembly
- Live register reports
- Resource usage CSV
- Screenshots of RGA output

---

## 5. What This Work Is

This is:

- Architecture-level kernel analysis
- Register pressure modeling
- Occupancy constraint quantification
- Hardware-aware attention redesign

This is not:

- A runtime benchmark suite
- A FlashAttention reimplementation
- A performance marketing comparison

All claims are grounded in compiler-generated VGPR allocation data.

See:

`docs/limitations_and_validation_scope.md`

---

## 6. Proposed Solution

Instead of persistent full HEAD_DIM accumulators per lane:

Wavefront-64 Native Design:

- Split HEAD_DIM across lanes
- Reduce per-lane accumulator count
- Introduce cross-lane reduction
- Maintain ≥2 waves per SIMD

This restores:

- Occupancy
- Latency hiding
- MFMA pipeline efficiency

Details:

`docs/wavefront64_native_attention_tile.md`

---

## 7. NVIDIA vs AMD Comparison

See:

`docs/nvidia_vs_wave64_register_model.md`

Summary:

| Feature | NVIDIA Warp-32 | AMD Wavefront-64 |
|----------|----------------|------------------|
| Execution width | 32 | 64 |
| Register allocation | Per thread | Per wave |
| Register limit | Large per SM | 256 per SIMD |
| Scaling behavior | Gradual | Hard collapse |

---

## 8. Research Positioning

This repository establishes:

- A register-level explanation for attention scaling limits on CDNA
- A wave-native tiling strategy
- A formal occupancy collapse threshold

It provides a foundation for:

- ROCm kernel redesign
- Triton backend tuning
- Vendor-aware attention optimization

---

## 9. Author Context

This project is part of an independent architectural study of:

Wavefront-64-native attention kernel design for AMD CDNA architectures.

All experiments were conducted using:

- Radeon GPU Analyzer (gfx950 target)
- ISA disassembly
- Compiler live register reports

---

## 10. Suggested Reading Order

1. `cdna4_wavefront64_register_model.md`
2. `flashattention3_under_cdna4_register_model.md`
3. `cdna4_occupancy_collapse_analysis.md`
4. `wavefront64_native_attention_tile.md`
5. `limitations_and_validation_scope.md`

---

## Closing Statement

This work argues that:

> Efficient attention kernels must be architecture-native.

FlashAttention is warp-native.  
CDNA requires wave-native.

Understanding that difference is the key to performance portability.

