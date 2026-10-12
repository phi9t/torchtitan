import TraceLifecycle

namespace Qwen3Formal.SingleRankFacts

open Qwen3Formal

def runId : String := "qfv-scout-a-seed42-fix1"
def attemptId : String := "single-rank-cuda-fix1"
def traceId : String := "sha256:e3e7d0c8f4eec80501e85fc201d7fe12c5a8298aeddbc9bc5d0432c2ebd005a7"

def observedEvents : List ObservedEvent := [
  {
    id := "qfv-scout-a-seed42-fix1:single-rank-cuda-fix1:r0:e000001"
    rawSourceId := "raw-r0-000001"
    kind := .stepStarted
    phase := .step
    predecessors := []
  },
  {
    id := "qfv-scout-a-seed42-fix1:single-rank-cuda-fix1:r0:e000002"
    rawSourceId := "raw-r0-000002"
    kind := .batchObserved
    phase := .input
    predecessors := ["qfv-scout-a-seed42-fix1:single-rank-cuda-fix1:r0:e000001"]
  },
  {
    id := "qfv-scout-a-seed42-fix1:single-rank-cuda-fix1:r0:e000003"
    rawSourceId := "raw-r0-000003"
    kind := .forwardStarted
    phase := .forward
    predecessors := ["qfv-scout-a-seed42-fix1:single-rank-cuda-fix1:r0:e000002"]
  },
  {
    id := "qfv-scout-a-seed42-fix1:single-rank-cuda-fix1:r0:e000004"
    rawSourceId := "raw-r0-000004"
    kind := .forwardCompleted
    phase := .forward
    predecessors := ["qfv-scout-a-seed42-fix1:single-rank-cuda-fix1:r0:e000003"]
  },
  {
    id := "qfv-scout-a-seed42-fix1:single-rank-cuda-fix1:r0:e000005"
    rawSourceId := "raw-r0-000005"
    kind := .backwardStarted
    phase := .backward
    predecessors := ["qfv-scout-a-seed42-fix1:single-rank-cuda-fix1:r0:e000004"]
  },
  {
    id := "qfv-scout-a-seed42-fix1:single-rank-cuda-fix1:r0:e000006"
    rawSourceId := "raw-r0-000006"
    kind := .gradientReady
    phase := .backward
    predecessors := ["qfv-scout-a-seed42-fix1:single-rank-cuda-fix1:r0:e000005"]
  },
  {
    id := "qfv-scout-a-seed42-fix1:single-rank-cuda-fix1:r0:e000007"
    rawSourceId := "raw-r0-000007"
    kind := .backwardCompleted
    phase := .backward
    predecessors := ["qfv-scout-a-seed42-fix1:single-rank-cuda-fix1:r0:e000006"]
  },
  {
    id := "qfv-scout-a-seed42-fix1:single-rank-cuda-fix1:r0:e000008"
    rawSourceId := "raw-r0-000008"
    kind := .optimizerStarted
    phase := .optimizer
    predecessors := ["qfv-scout-a-seed42-fix1:single-rank-cuda-fix1:r0:e000007"]
  },
  {
    id := "qfv-scout-a-seed42-fix1:single-rank-cuda-fix1:r0:e000009"
    rawSourceId := "raw-r0-000009"
    kind := .optimizerMutated
    phase := .optimizer
    predecessors := ["qfv-scout-a-seed42-fix1:single-rank-cuda-fix1:r0:e000008"]
  },
  {
    id := "qfv-scout-a-seed42-fix1:single-rank-cuda-fix1:r0:e000010"
    rawSourceId := "raw-r0-000010"
    kind := .stepCompleted
    phase := .step
    predecessors := ["qfv-scout-a-seed42-fix1:single-rank-cuda-fix1:r0:e000009"]
  },
]

end Qwen3Formal.SingleRankFacts
