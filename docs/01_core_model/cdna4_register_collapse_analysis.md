# CDNA4 Register Occupancy Collapse  
## Persistent MFMA Accumulator Study

---

## Abstract

This document analyzes the occupancy behavior of persistent MFMA accumulators
on CDNA4 (gfx950) GPUs using controlled scaling experiments.

We show that accumulator persistence directly induces
wavefront-64 occupancy collapse through VGPR pressure.

---

## Problem

Wavefront-64 architectures provide:

- 256 VGPR per SIMD
- Allocation granularity of 8
- Limited residency slots

Persistent MFMA accumulators remain live across multiple loop iterations.

As accumulator count increases, VGPR pressure grows.

The hypothesis:

Persistent accumulation scales linearly in VGPR footprint  
but reduces occupancy non-linearly.

---

## Methodology

We constructed synthetic OpenCL kernels:

- Identical MFMA instruction
- Controlled number of independent accumulators
- No shared memory usage
- Minimal control flow
- Identical loop structure

Accumulator counts swept:

1 → 2 → 4 → 8 → 16 → 32

VGPR usage extracted via Radeon GPU Analyzer.

---

## Results

| Accumulators | VGPR Used | Waves per SIMD |
|--------------|-----------|----------------|
| 1            | 16        | 16             |
| 2            | 16        | 16             |
| 4            | 16        | 16             |
| 8            | 80        | 3              |
| 16           | 136       | 1              |
| 32           | 296       | 0              |

---

## Analysis

### Phase 1 — Compiler Reuse Zone

Small accumulator counts allow:

- Aggressive VGPR reuse
- Flat register footprint
- Full occupancy

---

### Phase 2 — Structural Pressure Zone

At 8 accumulators:

- VGPR usage increases sharply
- Wave residency drops from 16 → 3
- Latency hiding begins to degrade

---

### Phase 3 — Collapse Zone

At 16 accumulators:

- Only 1 wave per SIMD possible
- Compute and memory latency cannot be hidden
- Throughput becomes pipeline-bound

---

### Phase 4 — Physical Limit

At 32 accumulators:

- VGPR requirement exceeds physical SIMD capacity
- Indicates spilling or allocation saturation

---

## Key Insight

Persistent MFMA accumulation is not occupancy-neutral.

On wavefront-64 architectures:

Accumulator lifetime amplification  
→ VGPR pressure amplification  
→ Non-linear occupancy collapse

---

## Implication for Attention Design

FlashAttention-style persistent accumulation must be:

- Blocked
- Staged
- Reduced
- Or lifetime-truncated

To remain hardware-aligned on CDNA4.

Wavefront-64-native attention design requires
register lifetime minimization as a first-class constraint.
