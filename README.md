# Architecture-Native Attention on AMD CDNA (Wavefront-64)

## TL;DR

FlashAttention-style kernels are optimized for NVIDIA Warp-32 execution.

AMD CDNA uses Wavefront-64 execution with **per-wave VGPR allocation (256 VGPR limit)**.

This mismatch causes **register pressure collapse at HEAD_DIM ≥ 128**, reducing occupancy to 1 wave per SIMD.

This repository:

- Quantifies the collapse using compiler-verified VGPR data (RGA)
- Builds a formal Wave-64 register scaling model
- Explains why persistent accumulators scale differently on CDNA
- Proposes a Wavefront-64-native alternative tile design

---

## The Core Insight

### NVIDIA (Warp-32)
- Per-thread register allocation
- Persistent accumulators scale linearly
- FlashAttention maps naturally

### AMD CDNA (Wave-64)
- Per-wave VGPR allocation (256 max)
- Accumulators scale across 64 lanes
- VGPR usage grows superlinearly with HEAD_DIM
- Occupancy collapses at 128 head dimension

> On CDNA, register allocation is a wave-level constraint, not a thread-level constraint.

---

## Quantified Example (gfx950 / CDNA4)

| HEAD_DIM | Used VGPR | Waves per SIMD |
|-----------|-----------|----------------|
| 32        | 80        | 3              |
| 64        | 136       | 1–2            |
| 128       | 296       | 1 (collapse)   |

Waves per SIMD = floor(256 / VGPR_per_wave)

At 296 VGPR:
- Exceeds 256-VGPR wave allocation window
- Scheduler restricted to 1 resident wave
- Latency hiding collapses

---

## What This Repo Contains

### 1. CDNA4 Register Model
- VGPR scaling derivation
- Occupancy heatmaps
- Spill boundary analysis
- ISA live-register validation

### 2. FlashAttention Under Wave-64
- Persistent accumulator scaling analysis
- HEAD_DIM sensitivity study
- Case study: BLOCK_M=128, BLOCK_N=64

### 3. Wavefront-64 Native Alternative
- Split-head mapping across lanes
- Accumulator lifetime reduction
- Cross-lane reduction strategy
- Occupancy-aware VGPR budgeting

### 4. Compiler-Verified Experiments
- Radeon GPU Analyzer outputs
- Live VGPR range inspection
- MFMA resource usage analysis

---

## Key Engineering Conclusions

1. Persistent accumulators scale per-lane on Wave-64.
2. HEAD_DIM ≥ 128 triggers occupancy collapse.
3. Architecture-native tiling restores multi-wave residency.
4. Register budgeting must precede kernel optimization.
5. FlashAttention-3 remains effective when HEAD_DIM ≤ 64 on CDNA.

---

## Why This Matters

Kernel optimizations are not architecture-agnostic.

Warp-optimized persistent kernels do not automatically map efficiently to Wave-64 execution.

Architecture-native design is required to:

- Sustain multi-wave occupancy
- Avoid register spills
- Preserve latency hiding
- Maintain scalable performance

---

## Intended Audience

- GPU kernel engineers
- ML systems engineers
- Compiler engineers
- Architecture performance teams

---

## Repository Structure
docs/
├── 01_core_model
├── 02_flashattention_analysis
├── 03_wave64_native_design
├── 04_experiments
├── 05_comparative_analysis
└── 06_whitepaper


Start here:

→ `docs/00_project_overview.md`

---

## Status

This is an architectural and compiler-level study.

- No runtime CDNA benchmarking yet
- All register claims verified via RGA
- Designed for hardware-aware kernel engineers

---

## Author

Chandan Kumar  
C++ / GPU Architecture / Kernel Optimization  
Focused on architecture-native ML kernel design