# Progress
- 2026-10-15 ~05:00Z opened from owner request. A01 launched (Steps 1-3 + all-arm smoke). A02 pilot gated on A01.
- 2026-10-15 ~06:05Z A01 report written. Geometry tests and runner sanity pass, but A01 is not green: Waystone/FineWeb verified cache is unavailable, GPUs are shared/crowded, and the all-arm dry-run, per-arm overfit/lr=0 controls, recall-after-train, and anchor 5-minute gate remain blocked. A02 remains blocked.
- 2026-10-15 ~16:20Z A01b launched (brief `sdd/a01b-brief.md`): data gate via host-side stage.sh/hdfs_strict.sh on the
  S07b-staged FineWeb ecbb586 cache (md5 89e8165a...); GPU 7 exclusive; G1 all-arm dry-run divergence, G2 overfit + lr=0,
  G3 5-min anchor. A02 still gated.
- 2026-10-15 ~16:40Z owner: "try really hard to get them to work". Spec Revision 2 (before any A02 data): 4x3 LR/init
  screen per arm (36 + 2 anchor), fixed rescue menu, all 4 arms x 3 seeds confirmation, behavior diagnostics. A02 brief
  written; launches the moment A01b is green, on every free GPU.
- 2026-10-15 ~17:35Z owner expanded scope to four families (A similarity kernels, B normalization/selection, C
  finite-feature, D stateful memories), separate leaderboards. Spec Revision 3 records Family A (14 kernels) with an
  implementation tier per arm (SDPA / flex score_mod / explicit side table). A03 (Family A kernel library + tests, new
  files only) launched; B-D await owner detail.
- 2026-10-15 ~18:08Z A01b report written. A01b is green: host-verified FineWeb cache, G1 all-arm 50-step FineWeb
  divergence, G2 fixed-bank overfit plus lr=0 controls for all arms, recall-after-train execution, and G3 5-minute
  anchor timing all passed on GPU 7. A02 precondition is now satisfied; use spec Revision 2 / `sdd/a02-brief.md`, not
  the older 19-run pilot.
