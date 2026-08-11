# Countdown Split Registry Before Expanded Claims

The Countdown scaffold-to-policy work will use a canonical split registry with
global `Problem Key` de-duplication before making expanded scientific claims.
The completed pilot showed strong OOD gains but also exposed exact train/dev/IID
overlap from seed reuse, so future Countdown claims must pass an evidence ladder:
runtime preflight, split-overlap validation, base metrics, adapter metrics,
base-elicitable subset metrics, verifier failure breakdown, and representative
examples. We accept the extra setup cost because broader coding and reasoning
benchmarks are only meaningful after the core held-out evidence contract is
reliable.
