# FlashAttention vs Wavefront-64  
## A Quantitative Mismatch Analysis (CDNA4 – gfx950)

---

## Executive Summary

On CDNA4 (gfx950):

- VGPRs per SIMD: **256**
- Empirically derived register model:

\[
R(A) \approx 16 + 8A
\]

Where:

- `A` = number of persistent MFMA accumulator fragments per thread
- `R(A)` = total VGPR usage per thread

Two-wave safety condition:

\[
\left\lfloor \frac{256}{R(A)} \right\rfloor \ge 2
\]

Solving:

\[
16 + 8A \le 128
\]

\[
A \le 14
\]

**Safe persistent accumulator threshold on CDNA4 ≈ 14 fragments per thread**

FlashAttention-style tiling often exceeds this.

Result:

> FlashAttention’s register residency model structurally conflicts with wavefront-64 register economics.

This is not a compiler issue.  
It is an architectural mismatch.

---

## 1. CDNA4 Register Model (Derived from RGA Sweep)

From empirical MFMA occupancy sweep experiments:

| Accumulators (A) | VGPR Used |
|------------------|-----------|
| 1                | 16        |
| 2                | 16        |
| 4                | 16        |
| 8                | 80        |
| 16               | 136       |
| 32               | 296       |

For A ≥ 8, the growth is approximately linear:

\[
R(A) \approx 16 + 8A
\]

Where:

- 16 VGPR ≈ baseline loop / address / bookkeeping
- 8 VGPR per accumulator fragment

This slope was derived directly from:

- `gfx950_sweep_*_resourceUsage.csv`

---

## 2. Occupancy Implications

Given:

- VGPR per SIMD = 256
- Wavefront size = 64

Theoretical waves per SIMD:

\[
W(A) = \left\lfloor \frac{256}{R(A)} \right\rfloor
\]

Examples:

| A | R(A) | Waves |
|---|------|-------|
| 8  | 80  | 3 |
| 12 | 112 | 2 |
| 14 | 128 | 2 |
| 16 | 136 | 1 |
| 32 | 296 | 0 (spill required) |

**Collapse point begins at A ≈ 15–16**

Beyond this, occupancy drops to **1 wave per SIMD**.

---

## 3. What FlashAttention Requires

Typical FlashAttention tile (example: 128×64):

Per thread resources include:

| Component | Estimated VGPR |
|-----------|----------------|
| Accumulators (8 fragments) | 64 |
| Q fragment | 16 |
| K fragment | 16 |
| V fragment | 16 |
| Softmax temps | 8–16 |
| Address + loop temps | 8–16 |
| Double buffering | 16–32 |

Conservative total:

\[
\approx 140–180\ VGPR
\]

Using register model:

\[
W = \left\lfloor \frac{256}{150} \right\rfloor = 1
\]

This implies:

> FlashAttention naturally collapses to single-wave occupancy on wavefront-64 hardware.

---

## 4. Why This Was Acceptable on CUDA

FlashAttention was tuned for:

- Warp size = 32
- Different register file partitioning
- Tensor core fragment mapping per warp
- Distinct occupancy vs latency tradeoffs

On wavefront-64:

- Per-thread register footprint effectively doubles
- Accumulator persistence scales linearly
- Register pressure compounds across 64 lanes

FlashAttention optimizes memory bandwidth.  
Wavefront-64 penalizes register residency.

The optimization target shifts.

---

## 5. Structural Mismatch

FlashAttention assumes:

- Long-lived accumulators
- Large fragment concurrency
- Persistent MMA fragments
- High tile reuse

Wavefront-64 favors:

- Short accumulator lifetime
- Fragment staging
- Higher wave multiplicity
- Register reuse over persistence

Therefore:

> FlashAttention’s fragment residency model conflicts with CDNA4’s wavefront-64 occupancy constraints.

---

## 6. Experimental Validation

This repository demonstrates:

- MFMA accumulator scaling slope ≈ 8 VGPR per fragment
- Safe threshold ≈ 14 persistent fragments
- Occupancy collapse at A ≥ 16
- Scratch spilling when exceeding hardware limit (A ≥ 32)

This establishes a measurable architectural boundary.

---

## 7. Practical Consequences

To make FlashAttention-like kernels efficient on CDNA4:

- Reduce concurrent accumulator fragments
- Shorten accumulator lifetime
- Avoid excessive double buffering in registers
- Stage softmax differently
- Retile for wavefront-64 semantics

This motivates:

> Wavefront-64 Native Attention Design

---

## 8. Conclusion

FlashAttention is not inefficient.

It is optimized for a different register economy.

On CDNA4:

- Persistent accumulator depth beyond ~14 fragments per thread
- Forces occupancy collapse
- Reduces latency hiding
- Risks spilling under realistic workloads

This is a hardware-constrained algorithmic mismatch.

Understanding this boundary enables:

- Architecture-aware kernel redesign
- Proper tiling strategies
- CDNA-native attention implementations
