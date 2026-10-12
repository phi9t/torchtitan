/-
EVALUATION OF THE OBSERVED TRACE, not a theorem about the protocol.

Every predicate here is `Bool`-valued, and SingleRankValid.lean applies it to
literal SingleRankFacts data and closes it with `rfl` or `decide`. What the kernel
checks is that THIS run satisfies the predicate: genuine, axiom-free, and
silent about any other trace. The result tokens say so, carrying
`kind=evaluation scope=observed-trace`.

For theorems quantified over topologies, states and schedule lengths -- which
mention no trace and no bound -- see DeviceMeshProtocol.lean,
DeviceMeshInductiveInvariant.lean and DeviceMeshWaitGraph.lean, whose tokens carry
`kind=theorem scope=all-topologies-all-schedules bound=none`.
-/

namespace Qwen3Formal

inductive EventKind where
  | stepStarted
  | batchObserved
  | forwardStarted
  | forwardCompleted
  | backwardStarted
  | gradientReady
  | gradientMissing
  | backwardCompleted
  | optimizerStarted
  | optimizerMutated
  | stepCompleted
deriving DecidableEq, Repr

inductive Phase where
  | step
  | input
  | forward
  | backward
  | optimizer
deriving DecidableEq, Repr

structure ObservedEvent where
  id : String
  rawSourceId : String
  kind : EventKind
  phase : Phase
  predecessors : List String
deriving DecidableEq, Repr

def beforeLoop (before after : EventKind) : Bool -> List ObservedEvent -> Bool
  | _, [] => true
  | seen, event :: rest =>
      if event.kind == after then
        seen
      else
        beforeLoop before after (seen || event.kind == before) rest

def occursBeforeIfPresent
    (events : List ObservedEvent)
    (before after : EventKind) : Bool :=
  beforeLoop before after false events

def forwardBeforeBackward (events : List ObservedEvent) : Bool :=
  occursBeforeIfPresent events .forwardStarted .forwardCompleted &&
  occursBeforeIfPresent events .forwardCompleted .backwardStarted &&
  occursBeforeIfPresent events .backwardStarted .backwardCompleted

def gradientReadyBeforeOptimizer (events : List ObservedEvent) : Bool :=
  occursBeforeIfPresent events .gradientReady .optimizerMutated

def optimizerBeforeStepEnd (events : List ObservedEvent) : Bool :=
  occursBeforeIfPresent events .optimizerMutated .stepCompleted

def predecessorsSeen : List String -> List String -> Bool
  | _, [] => true
  | seen, predecessor :: rest =>
      seen.contains predecessor && predecessorsSeen seen rest

def causalLoop : List String -> List ObservedEvent -> Bool
  | _, [] => true
  | seen, event :: rest =>
      predecessorsSeen seen event.predecessors &&
      causalLoop (event.id :: seen) rest

def causalOrder (events : List ObservedEvent) : Bool :=
  causalLoop [] events

def lifecycleValid (events : List ObservedEvent) : Bool :=
  forwardBeforeBackward events &&
  gradientReadyBeforeOptimizer events &&
  optimizerBeforeStepEnd events &&
  causalOrder events

end Qwen3Formal
