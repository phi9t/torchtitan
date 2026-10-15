# A02 brief: fair best-shot pilot (spec.md Revision 2) -- GATED on A01b green

Read: `spec.md` (Revision 2 is binding), `sdd/a01-report.md`, `sdd/a01b-report.md`. Reuse A01/A01b code and runner.
Precondition: A01b report says G1/G2/G3 PASS. If not, stop and report.

Before spending tokens:
- Implement the Revision-2 knobs if missing: separate AdamW group for geometry scalars, fp32 gauss key penalty (keep the
  unit tests green; add a test that the penalty is computed in fp32), geometry-scalar warmup knob, and the diagnostics
  (per-layer max |logit|, attention entropy on a fixed eval batch, s/tau/sigma, ||q||/||k||) logged at every eval into
  the attempt artifact. Diagnostics must not change training numerics (check: anchor 50-step loss identical with
  diagnostics on/off).
- One 50-step dry run per arm after these edits (arms must still diverge).

Execution: one run per GPU, exclusive. Use only GPUs with no foreign process (check nvidia-smi before each launch; never
touch processes you did not start). Fill free GPUs in parallel with a simple queue. Data: the host-verified FineWeb
cache as in A01b (hdfs_strict.sh --quick before each launch; record URI + md5).
Order: A02a screen (36 + 2 anchor) -> rescue round per spec -> A02b confirmation (12) + matched-token reruns + recall
probe. Freeze the per-arm best config from the screen table by the declared rule (lowest 5-min held-out loss; ties ->
lower LR scale) and write it into the report before launching A02b.

Report `sdd/a02-report.md`: screen table (all 38 + rescue), frozen configs, confirmation table with mean +/- spread per
arm, time-to-anchor-target, recall-probe results, per-arm behavior diagnosis (why an arm wins/loses: logit growth,
entropy collapse, temperature drift), claim labels (`representative_small`, `replicated_eval` only with 3 seeds).
Rootfs for all Python/CUDA. Do not commit.
