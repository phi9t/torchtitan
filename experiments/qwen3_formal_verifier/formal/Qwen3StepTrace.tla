------------------------------ MODULE Qwen3StepTrace ------------------------------
EXTENDS Naturals, FiniteSets

CONSTANTS Ranks, DPs, TPs, ProcessGroups, Works

Ranks == {0, 1, 2, 3}
DPs == {0, 1}
TPs == {0, 1}
ProcessGroups == {"tp_group_dp0", "tp_group_dp1", "dp_group_tp0", "dp_group_tp1"}
Works == {"w-step3-layer0-tp-attn-dp0", "w-step3-layer0-tp-attn-dp1", "w-step3-layer0-tp-mlp-dp0", "w-step3-layer0-tp-mlp-dp1", "w-step3-layer0-dp-grad-tp0", "w-step3-layer0-dp-grad-tp1"}

Axis == ("tp_group_dp0" :> "TP") @@ ("tp_group_dp1" :> "TP") @@ ("dp_group_tp0" :> "DP") @@ ("dp_group_tp1" :> "DP")
Members == ("tp_group_dp0" :> {0, 1}) @@ ("tp_group_dp1" :> {2, 3}) @@ ("dp_group_tp0" :> {0, 2}) @@ ("dp_group_tp1" :> {1, 3})
CoordDP == (0 :> 0) @@ (1 :> 0) @@ (2 :> 1) @@ (3 :> 1)
CoordTP == (0 :> 0) @@ (1 :> 1) @@ (2 :> 0) @@ (3 :> 1)
Coord == [r \in Ranks |-> [dp |-> CoordDP[r], tp |-> CoordTP[r]]]
WorkGroup == ("w-step3-layer0-tp-attn-dp0" :> "tp_group_dp0") @@ ("w-step3-layer0-tp-attn-dp1" :> "tp_group_dp1") @@ ("w-step3-layer0-tp-mlp-dp0" :> "tp_group_dp0") @@ ("w-step3-layer0-tp-mlp-dp1" :> "tp_group_dp1") @@ ("w-step3-layer0-dp-grad-tp0" :> "dp_group_tp0") @@ ("w-step3-layer0-dp-grad-tp1" :> "dp_group_tp1")
WorkRole == ("w-step3-layer0-tp-attn-dp0" :> "tp_attention_output_reduce") @@ ("w-step3-layer0-tp-attn-dp1" :> "tp_attention_output_reduce") @@ ("w-step3-layer0-tp-mlp-dp0" :> "tp_mlp_down_reduce") @@ ("w-step3-layer0-tp-mlp-dp1" :> "tp_mlp_down_reduce") @@ ("w-step3-layer0-dp-grad-tp0" :> "dp_gradient_sync") @@ ("w-step3-layer0-dp-grad-tp1" :> "dp_gradient_sync")
ProducerAten == ("w-step3-layer0-tp-attn-dp0" :> "aten-step3-layer0-rank0-attn-out") @@ ("w-step3-layer0-tp-attn-dp1" :> "aten-step3-layer0-rank2-attn-out") @@ ("w-step3-layer0-tp-mlp-dp0" :> "aten-step3-layer0-rank0-mlp-down") @@ ("w-step3-layer0-tp-mlp-dp1" :> "aten-step3-layer0-rank2-mlp-down") @@ ("w-step3-layer0-dp-grad-tp0" :> "qwen3-layer0-tp0") @@ ("w-step3-layer0-dp-grad-tp1" :> "qwen3-layer0-tp1")
WorkShardTP == ("w-step3-layer0-dp-grad-tp0" :> 0) @@ ("w-step3-layer0-dp-grad-tp1" :> 1)

DPGroupShape ==
  \A pg \in ProcessGroups :
    Axis[pg] = "DP" => \A r1, r2 \in Members[pg] : Coord[r1].tp = Coord[r2].tp

TPGroupShape ==
  \A pg \in ProcessGroups :
    Axis[pg] = "TP" => \A r1, r2 \in Members[pg] : Coord[r1].dp = Coord[r2].dp

DPShardAgreement ==
  \A w \in DOMAIN WorkShardTP :
    \A r \in Members[WorkGroup[w]] : Coord[r].tp = WorkShardTP[w]

=============================================================================
