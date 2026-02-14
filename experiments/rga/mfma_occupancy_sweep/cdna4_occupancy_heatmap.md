# CDNA4 Occupancy Heatmap (gfx950)
## Persistent MFMA Accumulator Sweep

---

## Hardware Limits

VGPR per SIMD: 256  
Wavefront size: 64  

Wave residency constraint:

waves = floor(256 / USED_VGPR)

---

## Measured Data

| Accumulators | Used VGPR | Waves per SIMD | Regime |
|--------------|-----------|----------------|--------|
| 1            | 16        | 16             | Compiler reuse |
| 2            | 16        | 16             | Compiler reuse |
| 4            | 16        | 16             | Compiler reuse |
| 8            | 80        | 3              | Structural pressure |
| 16           | 136       | 1              | Occupancy collapse |
| 32           | 296       | 0*             | Exceeds VGPR budget |

*Logical requirement exceeds 256 VGPR hardware limit.

---

## Regime Interpretation

### High Occupancy Zone (≥ 8 waves)
- Accumulators reused
- Minimal structural pressure
- Compiler dominates allocation behavior

### Transitional Zone (2–3 waves)
- True register scaling begins
- Accumulators independently live
- Latency hiding degrades

### Collapse Zone (≤ 1 wave)
- MFMA fully dominates VGPR budget
- No meaningful latency hiding
- Memory stalls become visible

---

## Derived Model

R_total(A) ≈ 16 + 8A

waves(A) = floor(256 / (16 + 8A))

Safe bound:

A ≤ 14 to maintain ≥ 2 waves.

---

## Key Insight

Wavefront-64 architectures impose a strict
accumulator residency budget.

Persistent MFMA-heavy kernels scale linearly in VGPR cost
once compiler reuse breaks.

This experimentally demonstrates the occupancy collapse boundary.
