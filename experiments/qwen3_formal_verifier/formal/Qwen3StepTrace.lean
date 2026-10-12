namespace Qwen3StepTrace

structure Coord where
  dp : Nat
  tp : Nat
deriving DecidableEq, Repr

structure Trace where
  ranks : List Nat
  coord : Nat -> Coord
  works : List String
deriving Repr

def qwen3TraceRanks : List Nat := [0, 1, 2, 3]
def qwen3TraceCoord : Nat -> Coord
| 0 => { dp := 0, tp := 0 }
| 1 => { dp := 0, tp := 1 }
| 2 => { dp := 1, tp := 0 }
| 3 => { dp := 1, tp := 1 }
| _ => { dp := 0, tp := 0 }
def qwen3TraceWorks : List String := ["w-step3-layer0-tp-attn-dp0", "w-step3-layer0-tp-attn-dp1", "w-step3-layer0-tp-mlp-dp0", "w-step3-layer0-tp-mlp-dp1", "w-step3-layer0-dp-grad-tp0", "w-step3-layer0-dp-grad-tp1"]
def qwen3TraceTorchOps : List String := ["aten-step3-layer0-rank0-attn-out", "aten-step3-layer0-rank0-mlp-down", "aten-step3-layer0-rank1-attn-out", "aten-step3-layer0-rank1-mlp-down", "aten-step3-layer0-rank2-attn-out", "aten-step3-layer0-rank2-mlp-down", "aten-step3-layer0-rank3-attn-out", "aten-step3-layer0-rank3-mlp-down"]

def sameTpCoord (tr : Trace) (rs : List Nat) : Prop :=
  forall r1, List.Mem r1 rs -> forall r2, List.Mem r2 rs -> (tr.coord r1).tp = (tr.coord r2).tp

def sameDpCoord (tr : Trace) (rs : List Nat) : Prop :=
  forall r1, List.Mem r1 rs -> forall r2, List.Mem r2 rs -> (tr.coord r1).dp = (tr.coord r2).dp

def meshWellFormed (_tr : Trace) : Prop :=
  True

def qwen3Trace : Trace :=
  { ranks := qwen3TraceRanks, coord := qwen3TraceCoord, works := qwen3TraceWorks }

end Qwen3StepTrace
