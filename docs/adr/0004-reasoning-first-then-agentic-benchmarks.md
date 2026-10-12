# Reasoning First, Then Agentic Benchmarks

The scaffold-to-policy expansion will proceed through a reasoning lane before
agentic task benchmarks. Countdown split repair remains the immediate priority;
after that, small exact-verifier reasoning benchmarks should come before
tau2-bench, Terminal-Bench, or Harbor-backed evaluations. We still want those
agentic benchmarks in scope, but they introduce external harnesses, tool/user or
terminal interaction semantics, and slower artifact loops, so they should enter
behind a stable experiment registry and held-out evidence ladder rather than be
mixed into the first repair step.
