# Superseded initial completion intent

Status: superseded by `../spec.md`

The session initially treated commit `366737c37` as an implementation shape to
complete directly. The implementation audit found that its TLA+ was not
TLC-executable, its Lean output was not compiled and contained a placeholder
well-formedness proposition, and its trace was synthetic.

The user then selected a two-phase program: sequential Scout A and Scout B
tracer bullets followed by progressive design and refinement. The durable spec
replaces the initial completion ticket. No implementation edits were retained
from the interrupted direct-completion attempt.

Reference decisions retained by the spec:

- Ferric Continuum's hermetic Bazel/TLC pattern is the TLA+ precedent.
- TorchLean commit `de3192d` is the Lean version and cache-policy precedent.
- Bazel owns formal toolchains and large caches outside the checkout.
- RoboRev uses `traecli` with `seed-evolving`.
