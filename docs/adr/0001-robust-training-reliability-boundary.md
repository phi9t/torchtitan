# Robust Training Reliability Boundary

Robust training work in TorchTitan should focus on training-semantics evidence,
degraded-progress evidence, recovery equivalence, and correctness validation,
while fleet-level quarantine, replacement, and health scoring remain
responsibilities of the surrounding training platform. This boundary keeps
TorchTitan from becoming a scheduler or fleet manager, while ensuring it emits
the evidence and validation hooks those systems need to act safely.
