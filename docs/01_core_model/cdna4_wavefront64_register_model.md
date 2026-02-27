# CDNA4 Wavefront-64 Register Model
## Empirical VGPR Scaling and Occupancy Derivation (gfx950)

---

## Objective

Derive a first-order architectural model for VGPR scaling and wave residency
as a function of persistent MFMA accumulator count on CDNA4 (gfx950).

This model is grounded in empirical measurements collected via
Radeon GPU Analyzer (RGA).

---

## Experimental Setup

Architecture: CDNA4  
ASIC: gfx950  
Wavefront size: 64  
Available VGPRs per SIMD: 256  
SGPRs: 102  

Kernel design:
- Persistent MFMA accumulators
- Fully unrolled inner MFMA loop
- No LDS usage
- No scratch usage
- Accumulators updated per iteration

---

## Measured VGPR Usage

| Accumulators (A) | Used VGPRs |
|------------------|------------|
| 1                | 16         |
| 2                | 16         |
| 4                | 16         |
| 8                | 80         |
| 16               | 136        |
| 32               | 296        |

---

## Observations

### Regime 1 — Compiler Reuse (A ≤ 4)

VGPR remains flat at 16.

The compiler:
- Reuses registers aggressively
- Collapses accumulator lifetimes
- Avoids structural pressure

This is not linear scaling.

---

### Regime 2 — Structural Register Scaling (8 ≤ A ≤ 16)

Now accumulators become independently live.

Empirical delta:

A = 8  
80 - 16 = 64 → 8 VGPR per accumulator

A = 16  
136 - 16 = 120 → 7.5 VGPR per accumulator

Approximation:

R_acc ≈ 8 VGPR per float4 accumulator

---

## Derived Register Model

Let:

A = number of persistent float4 accumulators  
R_base = baseline register cost (non-acc state)

Measured:

R_base ≈ 16  
R_acc ≈ 8

Therefore:

R_total(A) ≈ 16 + 8A

---

## Wave Residency Model

Wave residency per SIMD is bounded by:

waves_per_SIMD(A) = floor(256 / R_total(A))

Substituting:

waves_per_SIMD(A) = floor(256 / (16 + 8A))

---

## Model Validation

A = 8  
16 + 64 = 80  
256 / 80 = 3 waves ✔

A = 16  
16 + 128 = 144  
256 / 144 = 1 wave ✔ (measured 136 VGPR → 1 wave)

A = 32  
16 + 256 = 272  
> 256 → exceeds hardware capacity ✔

Model closely matches empirical data.

---

## Safe Accumulator Threshold

To maintain ≥ 2 waves per SIMD:

16 + 8A ≤ 128  
8A ≤ 112  
A ≤ 14

Safe persistent accumulator bound:

A_safe ≈ 14

Beyond this, occupancy collapses to a single wave.

---

## Architectural Implication

Persistent accumulator designs (e.g., FlashAttention-style)
must remain under ~14 concurrent float4 accumulators
to avoid wave residency collapse on CDNA4 wavefront-64.

This model provides a concrete register budget constraint
for wavefront-native attention kernel design.
