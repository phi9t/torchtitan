# Implementation-brief and review checklist

Across the formalization backlog, reviews kept finding the same small number of
defect classes, and briefs kept asserting mechanisms nobody had measured. Both
lists below are empirical: every entry is something that actually happened in
this program, not a generic code-review checklist.

Use part 1 when writing a brief and when reviewing the result. Use part 2 before
asserting anything about existing code. Use part 3 when handing work to a
subagent.

## Part 1: defect classes reviews keep finding

Each is a question, with the evidence that answers it. "I read it and it looks
right" answers none of them.

1. **Can this guard fail?** Mutate the subject, re-run the named test, confirm
   the test fails, then restore and confirm it passes. Five guards in this
   program were written in a shape that could not fail: a test that grepped a
   script instead of running it; a skip detector matching text that colour codes
   had already mangled; a coverage ledger keyed on a name at invocation; a
   refusal test asserting only a non-zero exit; a label test enshrining an
   overclaim. Every one was found by mutation, and only because a reviewer chose
   to try it. Register the killing mutation rather than describing it.

2. **Is the antecedent satisfiable?** An invariant whose precondition is
   unreachable passes vacuously and looks identical to one that holds. Show a
   reachable state that satisfies the antecedent, or show the state count
   changing when the conjunct is removed.

3. **Is the negative refused at the intended guard?** A negative that exits
   non-zero for the wrong reason is not evidence. Assert the guard's name or
   message, not the exit status. The refinement bridge's first refutation target
   was satisfiable by issuing everything and running nothing, so both the
   positive and the negative would have passed without checking anything.

4. **Is configuration order pinned?** Unpinned cfg order masks which invariant
   was actually evaluated, so a passing run cannot be attributed.

5. **Does the result token claim only what the run covered?** A token reporting
   `bound=none` for a bounded run, or a suite label wider than the checks that
   ran, converts a partial result into a false one. The token is the artifact
   someone else reads; it carries the scope, not the intent.

6. **Is a digest recomputed, or only compared?** Comparing a stored digest to
   itself proves storage, not integrity. The recomputation is the check.

7. **Has anyone run the tool whose behaviour the brief asserts?** If not, the
   assertion is a hypothesis and must be labelled as one.

## Part 2: what a brief may not assert without a command

Six briefs in this program asserted a mechanism that measurement later refuted.
The shapes recur:

- "X is load-bearing" -- refuted by removing X and getting bit-identical
  results.
- "no X exists in this repository" -- refuted because the search covered two
  directories and the conclusion covered the repository. Twice.
- "the check fires at Y" -- refuted because the guard never reads the field the
  claim depends on.
- "this stage cannot run under condition Z" -- refuted by running it.

The rule that follows: any claim about what code does, what exists, or where a
check fires carries the command that produced it, or is marked explicitly as a
hypothesis for the implementer to test first. A brief is allowed to be
uncertain. It is not allowed to be confidently unmeasured.

Scope claims need the widest search that supports them. `grep` over two
directories supports a statement about two directories.

## Part 3: what made subagent execution work

Keep these three, which produced the useful results:

- latitude on design, with firm acceptance criteria;
- explicit permission to answer "your instruction is wrong, here is the
  measurement" -- used, and correct, more than once;
- the brief names the exact seam as `file:line` rather than describing it, so
  the implementer starts from the code instead of from a paraphrase.

## Operational rules that cost something to learn

- **Do not edit any file while a gate runs.** The source snapshot covers the
  whole tree, so editing anything -- including a ticket -- moves it and the
  recheck refuses the run. This cost a 20-minute GPU run.
- **A passing gate is not evidence of detection.** It shows the properties hold
  on the run that was made. Detection claims need an injected fault and a token
  naming the check that fired.
- **Smoke runs prove plumbing.** Nothing more, whatever else the log says.
