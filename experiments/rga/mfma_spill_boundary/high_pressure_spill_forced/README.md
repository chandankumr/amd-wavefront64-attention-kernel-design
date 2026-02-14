Spill threshold experiment on gfx950 (CDNA4).

Result:
- VGPR: 132
- Scratch: 1016 bytes
- Occupancy: 1 wave
- Heavy scratch_load/store traffic around MFMA blocks

Conclusion:
Spill occurs before raw VGPR limit due to persistent AGPR + VGPR lifetime overlap.
