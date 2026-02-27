# Architecture-Native Attention on AMD CDNA (Wavefront-64)

## Executive Summary

This project investigates a fundamental architectural mismatch:

> FlashAttention-style kernels were designed around NVIDIA Warp-32 execution and do not scale efficiently under AMD CDNA Wavefront-64 register allocation.

Using compiler-verified analysis (Radeon GPU Analyzer), this repository:

- Builds a quantitative VGPR register model for CDNA4 (gfx950)
- Demonstrates occupancy collapse at HEAD_DIM ≥ 128
- Explains why persistent accumulator models scale differently on Wave-64
- Proposes a Wavefront-64-native alternative tile design
- Validates register pressure behavior using ISA-level inspection

This is not an implementation repo.  
It is an architectural analysis and kernel design study.

---

## Problem Statement

FlashAttention relies on:

- Persistent accumulators
- Per-thread register scaling
- Warp-32 execution assumptions

On AMD CDNA:

- Execution width is Wavefront-64
- VGPR allocation is per-wave
- Each lane inherits accumulator growth
- Register usage scales differently

This creates a nonlinear increase in VGPR usage as HEAD_DIM increases.

At HEAD_DIM = 128:

- Accumulators dominate register file
- VGPR usage exceeds safe occupancy thresholds
- Wave-level occupancy collapses to 1 wave per SIMD

This is an architectural scaling issue — not a compiler bug.

---

## Core Hypothesis

> Attention kernels must be architecture-native.

A kernel optimized for Warp-32 execution does not automatically map efficiently to Wavefront-64 execution.

Instead of forcing FlashAttention into CDNA,
we explore a design where:

- Head dimension work is partitioned across lanes
- Accumulator lifetime is reduced
- VGPR pressure remains bounded
- Occupancy stays in the 2–3 wave safe zone

---

## What This Repository Contains

### 1. CDNA4 Register Model

A formal VGPR scaling model derived from:

- MFMA instruction behavior
- Accumulator lifetimes
- Compiler resource usage reports
- RGA live register analysis

Includes:

- Occupancy heatmaps
- Spill boundary analysis
- Accumulator lifetime experiments

---

### 2. FlashAttention Under Wave-64

Quantitative analysis of:

- Persistent accumulator scaling
- HEAD_DIM impact (32, 64, 128)
- VGPR consumption trends
- Occupancy collapse behavior

Case study:
BLOCK_M = 128  
BLOCK_N = 64  
HEAD_DIM = 128  

---

### 3. Wavefront-64 Native Alternative

Proposed design principles:

- Split-head partitioning across lanes
- Controlled accumulator lifetime
- Cross-lane reduction instead of per-thread accumulation
- Occupancy-aware register budgeting

Includes:

- Tile diagrams
- Pseudocode
- Performance projections
- Design contract

---

### 4. ISA-Level Experimental Validation

All major claims are supported by:

- Radeon GPU Analyzer outputs
- VGPR live-range dumps
- Resource usage CSV reports
- Controlled accumulator experiments

No speculative claims are made without compiler evidence.

---

## Key Findings

1. Persistent accumulators scale per-lane on Wave-64.
2. VGPR usage grows superlinearly with HEAD_DIM.
3. HEAD_DIM ≥ 128 triggers occupancy collapse.
4. Split-head mapping reduces per-lane VGPR pressure by ~4×.
5. Architecture-native tiling restores multi-wave occupancy.

---

## Limitations

- Analysis is based on compiler resource inspection (RGA).
- No runtime benchmarking on physical CDNA hardware.
- FP8 MFMA path is discussed but not empirically measured.
- FlashAttention-3 improvements are considered, but not reimplemented.

See: `limitations_and_validation_scope.md`

---

## Intended Audience

This repository is intended for:

- GPU compiler engineers
- Kernel optimization engineers
- ML systems researchers
- Architecture performance teams

It is written from a hardware-first perspective.

---

## Research Implication

The main architectural insight:

> Warp-optimized persistent kernels are not universally portable across execution models.

For CDNA Wavefront-64:

- Register budgeting must be wave-aware.
- Head dimension partitioning is critical.
- Accumulator lifetime control is mandatory.

Architecture-native kernel design is not optional at scale.

---

## Future Work

- Runtime microbenchmark validation on CDNA hardware
- FP8 scaled-MFMA modeling
- Compiler-assisted wave-aware tiling heuristics
- Automated VGPR budgeting tools

---

## Repository Philosophy

This project prioritizes:

- Quantitative modeling over intuition
- ISA-level evidence over assumptions
- Architecture-first reasoning
- Explicit tradeoff documentation

The goal is clarity, not hype.

---