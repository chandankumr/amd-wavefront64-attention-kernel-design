# Wavefront-64 Native Attention Kernel Design (AMD)

This repository is a **hardware-first analysis and design study** of attention kernels
on **AMD wavefront-64 GPUs (CDNA / MI-series)**.

It is written from the perspective of **GPU architecture constraints**, not framework
or algorithm preferences.

The goal is to answer one question:

> How should attention kernels be designed when wavefront-64 execution,
> split register files, and MFMA-based matrix compute are treated as first-class realities?

---

## Why This Repository Exists

Most modern attention kernels (e.g., FlashAttention) are designed around assumptions that hold well on:
- Warp-32 execution
- Unified register files
- Gradual occupancy degradation

AMD GPUs differ fundamentally:
- Wavefront-64 execution
- Physically split register files (ArchVGPR / AccVGPR)
- Sharp occupancy cliffs
- Explicit MFMA ↔ ALU domain crossings

This repository documents **where those assumptions break**, **when they still hold**, and
**how a wavefront-64-native design can be constructed instead**.

This is not a benchmark repo.
This is not a framework comparison.
This is an **architectural design study**.

---

## Structure of the Analysis

The repository is organized as a sequence of design reasoning steps.

### 1. Hardware Reality (ISA-Level)
- `isa_level_attention_mapping.md`  
  Physical constraints imposed by MFMA, AccVGPRs, and wavefront-64 execution.

### 2. Architectural Contracts
- `attention_kernel_contract_for_wavefront64.md`  
  Non-negotiable execution constraints any high-performance kernel must respect.

### 3. Failure Analysis
- `where_flashattention_violates_wavefront64_contract.md`  
  Where and why FlashAttention breaks those contracts on AMD GPUs.

### 4. Judgment, Not Dogma
- `when_flashattention_is_still_effective_on_wavefront64.md`  
  Specific regimes where FlashAttention remains valid and competitive.

### 5. Engineering Priorities
- `which_flashattention_violations_are_worth_fixing.md`  
  Which issues justify fixes, and which should be accepted or worked around.

### 6. Native Design Proposal
- `designing_wavefront64_native_attention.md`  
  A ground-up attention kernel architecture designed specifically for wavefront-64 GPUs.

---

## What This Repository Is (and Is Not)

**This repository is:**
- Architecture-driven
- Constraint-aware
- Focused on correctness, predictability, and performance tradeoffs

**This repository is not:**
- A claim that FlashAttention is “bad”
- A CUDA vs ROCm benchmark comparison
- An implementation drop-in replacement

---

## Intended Audience

This work is intended for:
- GPU kernel engineers
- Compiler engineers
- AI runtime / systems engineers
- Architecture-aware ML practitioners

---

## Status

This repository represents a **completed design investigation**.

Future work could include:
- Prototype kernels
- Cost-model-driven kernel selection
- Integration into runtime heuristics

Those are intentionally left out to keep the focus on **architecture and design clarity**.
