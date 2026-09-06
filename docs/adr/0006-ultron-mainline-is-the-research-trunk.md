# Ultron/mainline is the research trunk

Local `main` grew 946 unpublished commits on top of a February
`origin/main` that still looked like a pytorch/torchtitan snapshot. That
made the public default branch a lie and left `ultron/mainline` as a
stale worktree alias of yesterday's tip.

The research trunk is `ultron/mainline`, locally and on `origin`. Topic
branches merge back with fast-forward. GitHub's default branch is
`ultron/mainline`. `main` is deprecated on both sides and is not the
integration line.

We publish the existing local history rather than squash or orphan it:
the experiment programs are not understandable without the rootfs, BPF,
and research-vehicle commits they sit on. Production-release sanitization
(stripping machine-local paths and unrelated training work for a Vaso
candidate) is a later program and is not a gate on this trunk.

Considered and rejected: leaving `origin/main` as the public default
(keeps the fork looking like pytorch while the real work is hidden);
parking experiments on permanent topic branches (hides the trunk);
waiting for the sanitization campaign before any push (leaves the only
copy of Falcon and Mini Kimi on one disk).
