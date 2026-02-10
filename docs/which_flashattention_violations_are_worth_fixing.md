# Which FlashAttention Violations Are Worth Fixing on Wavefront-64 GPUs

This document evaluates **which architectural violations of FlashAttention are worth addressing**
on **AMD wavefront-64 GPUs**, and which should be **explicitly accepted or worked around**.

The goal is not to “make FlashAttention perfect on AMD,” but to apply **engineering judgment**
under real constraints: complexity, risk, and return on investment.

---

## 1. Why This Question Matters

FlashAttention violates several wavefront-64 architectural contracts:

- Long-lived accumulators
- Online softmax dependencies
- High register pressure
- Occupancy collapse

However:

> **Not every violation is worth fixing.**

Some violations:
- Are fundamental to the algorithm
- Would require a full redesign
- Offer poor performance return per engineering cost

This document distinguishes:
- **Fixable violations**
- **Acceptable violations**
- **Violations that justify alternative kernels**

---

## 2. Evaluation Criteria

Each violation is evaluated using four criteria:

| Criterion | Question |
|--------|---------|
| Performance Gain | Does fixing it recover meaningful performance? |
| Architectural Alignment | Does the fix respect wavefront-64 physics? |
| Engineering Cost | How invasive is the change? |
| Risk | Does it threaten correctness or stability? |

Only violations with **high gain and manageable cost** are worth fixing.

---

## 3. Violation: Long-Lived Output Accumulator

### Description

- FlashAttention keeps the output accumulator (`acc`) live across all KV tiles
- Accumulator is FP32 and large (BLOCK_M × HEAD_DIM)
- Lives in AccVGPRs and cannot be spilled cheaply

### Impact

- Dominant contributor to register pressure
- Triggers occupancy cliffs on wavefront-64
- Amplifies all other violations

### Fixability

✅ **Worth fixing**

### Why

- Accumulator lifetime can be shortened **without changing math**
- Split-K and partial writeback are viable
- Trade bandwidth for occupancy — favorable on AMD

### Recommendation

> **Prioritize accumulator lifetime reduction.**

This is the **highest ROI fix**.

---

## 4. Violation: Online Softmax Dependency Chain

### Description

- Running max and denominator persist across KV tiles
- Forces accumulator rescaling
- Prevents early writeback

### Impact

- Extends both ArchVGPR and AccVGPR live ranges
- Forces repeated domain crossings
- Limits scheduling flexibility

### Fixability

⚠️ **Conditionally worth fixing**

### Why

- Two-phase softmax removes dependency
- But requires extra memory passes
- Kernel launch overhead increases

### Recommendation

> Fix **only for long sequences or inference-critical paths**.

Do **not** replace online softmax universally.

---

## 5. Violation: Large Tile Sizes Optimized for NVIDIA

### Description

- Default FlashAttention tiles assume warp-32 economics
- Large tiles inflate register usage quadratically

### Impact

- Accelerates occupancy collapse
- Makes tuning fragile

### Fixability

✅ **Worth fixing**

### Why

- Tile sizes are tunable
- Smaller tiles degrade gracefully
- No algorithmic changes required

### Recommendation

> Provide **AMD-specific tile caps** and autotune constraints.

Low risk, immediate benefit.

---

## 6. Violation: MFMA ↔ Softmax Domain Crossing

### Description

- MFMA accumulates in AccVGPRs
- Softmax requires ArchVGPRs
- `v_accvgpr_read` has non-zero latency

### Impact

- Scheduling pressure
- MFMA pipeline stalls if occupancy is low

### Fixability

❌ **Not worth fixing**

### Why

- This is a **hardware-enforced constraint**
- Cannot be optimized away
- Only mitigated via occupancy and pipelining

### Recommendation

> Accept this violation. Design around it.

---

## 7. Violation: Reduction Cost on Wavefront-64

### Description

- Reductions require more steps than warp-32
- LDS usage scales with wave width

### Impact

- Higher fixed cost per softmax
- More sensitive to bank conflicts

### Fixability

❌ **Not worth fixing inside FlashAttention**

### Why

- Reduction topology is architectural
- Fix requires rethinking softmax itself
- Better addressed by alternative kernels

### Recommendation

> Do not optimize reductions inside FlashAttention.
> Choose different kernels when reductions dominate.

---

## 8. Violation: Low Occupancy Tolerance

### Description

- FlashAttention assumes performance degrades gradually
- On AMD, occupancy cliffs are sharp

### Impact

- Performance collapses unexpectedly
- Autotuning becomes unreliable

### Fixability

⚠️ **Indirectly fixable**

### Why

- Cannot change occupancy physics
- Can enforce selection heuristics

### Recommendation

> Fix at **dispatch level**, not kernel level.

Kernel selection heuristics are the correct solution.

---

## 9. Summary Table

| Violation | Worth Fixing? | Action |
|--------|--------------|-------|
| Long-lived accumulator | ✅ Yes | Split-K / writeback |
| Online softmax | ⚠️ Sometimes | Two-phase softmax |
| Large tiles | ✅ Yes | AMD-specific tiling |
| MFMA ↔ ALU crossing | ❌ No | Accept |
| Reduction cost | ❌ No | Use other kernels |
| Occupancy cliffs | ⚠️ Indirect | Selection heuristics |

---

## 10. Strategic Conclusion

FlashAttention should **not** be fully “ported” to AMD.

Instead:

- Fix **structural issues with high ROI**
- Accept **hardware-imposed constraints**
- Use **kernel selection** as the primary optimization lever

This approach minimizes risk while maximizing real-world performance.

---

## 11. Engineering Takeaway

Senior optimization is not about fixing everything.

It is about knowing:

> **What to fix, what to tolerate, and when to switch strategies.**

This distinction is what separates kernel hackers from GPU architects.

---

## Status

This document completes the **decision phase**.

Next steps naturally include:

- Designing a wavefront-64-native attention kernel
- Encoding selection heuristics into runtime logic
- Prototyping only the fixes that justify their cost
