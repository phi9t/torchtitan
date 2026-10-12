#!/usr/bin/env bash
# Bridge the DPxTP collective protocol model to the observed four-rank run.
#
# Ten checks. DeviceMeshRefine.tla's claim inventory carries the same list and the
# two must agree.
#
#   parse        how long the 1.6 MB generated facts module costs to parse and
#                semantically process. That cost is charged before a single
#                state is generated, so it is reported rather than folded into
#                a search time;
#   mapping      the observed-to-model communicator mapping is derivable,
#                bijective, and agrees with the model on member sets and
#                admissible operations, and the derived per-rank issue
#                sequences are orderable, equal in length, and long enough for
#                the replay bound. The derived mapping is printed;
#   refinement   the observed issue order of all four ranks, all 108 issues
#                each, runs to completion under every guard switched on. Read
#                what that does and does not say under WHAT THE POSITIVE
#                ESTABLISHES in DeviceMeshRefine.tla: the columns are typable and
#                cross-rank consistent and the collectives drain, which is not
#                a statement about absolute order;
#   order pair   and the limit is checked rather than described. A permutation
#                applied uniformly to all four ranks is ADMITTED, because the
#                protocol constrains cross-rank agreement and not absolute
#                order; the same permutation applied to ONE rank breaks that
#                agreement and is REFUSED;
#   confluence   over a declared fragment -- the first MaxIssues issues of each
#                rank, a per-rank issue spread of at most MaxReplaySkew and at
#                most MaxOutstanding issues in flight per rank and
#                communicator -- EVERY schedule reaches completion, checked
#                exhaustively through TLC's own deadlock detection;
#   overlap      and that fragment is not a serialized space: it contains a
#                state with two collectives in flight at once. A serialized
#                schedule satisfies every guard trivially, so a confluence
#                claim over a serialized space would be worth little;
#   negative     transposing two adjacent issues of one rank on one
#                communicator, with differing operations, is refused, the
#                mismatched collective never runs, and the refusal is no
#                earlier than that rendezvous -- the reach witness requires the
#                corrupted record to have been ISSUED, so a run refused at the
#                SPMD-program guard cannot satisfy it;
#   isolation    relaxing RequireMatchedIssueOrder alone makes that same
#                corruption complete, which attributes the refusal to NCCL's
#                matching requirement rather than to the stream-head guard;
#   guard set    and under the POSITIVE's guard set the same corruption is
#                refused too, differently: no peer ever reaches the corrupted
#                position, so the corrupted column never forms and the refuser
#                is the SPMD-program guard. A single-rank corruption is an
#                agreement violation and both guards are entitled to refuse
#                it; this names which one does when.
#
# Refinement polarity: TLC proves reachability by refutation, so the witness
# that the observed run is admitted is a violation of
# ReplayedRunIsNotAdmitted. This script translates that into a positive result
# token so the evidence does not read backwards. Same convention as
# run_tlc_single_rank_model.sh.
#
# WHY -Xss IS RAISED. DeviceMeshModel's SPMD-program guards contain
#
#   \A r1 \in Ranks : \A r2 \in Ranks : \A k \in 1..MinOf(len1, len2) : ...
#
# inside IssueAllowed, which is evaluated in action position, where TLC
# recurses once per bound element while building its action-item list. At
# MaxIssues = 2, where every other configuration of that model runs, the
# innermost range has two elements. Replaying 108 issues per rank makes it
# 4 x 4 x 108, and TLC overflows the default 1 MB thread stack at around
# replay depth 258 with a StackOverflowError -- which the checker contract
# correctly classifies as an infrastructure error rather than a result.
# Measured: the full replay completes at -Xss8m, and this runs at -Xss32m for
# margin. Attribution was verified by rerunning the same configuration with
# RequireUniformProgramOps and RequireUniformProgramComms relaxed, which
# completes the whole 865-step replay at the default stack size.

set -euo pipefail

[[ $# -eq 2 ]] || {
  echo "usage: run_tlc_device_mesh_refine.sh <java> <tla2tools.jar>" >&2
  exit 2
}

resolve_runfile() {
  local requested="$1"
  local candidate
  for candidate in \
    "${requested}" \
    "${TEST_SRCDIR:-}/${requested}" \
    "${TEST_SRCDIR:-}/${requested#external/}"; do
    if [[ -f "${candidate}" ]]; then
      readlink -f -- "${candidate}"
      return 0
    fi
  done
  echo "runfile not found: ${requested}" >&2
  return 1
}

workspace="${TEST_WORKSPACE:-_main}"
fixture_dir="${TEST_SRCDIR}/${workspace}/experiments/qwen3_formal_verifier/formal"
[[ -d "${fixture_dir}" ]] || {
  echo "Device-mesh formal fixtures not found: ${fixture_dir}" >&2
  exit 2
}
# shellcheck source=experiments/qwen3_formal_verifier/formal/checker_contract.sh
source "${fixture_dir}/checker_contract.sh"

java_bin="$(resolve_runfile "$1")"
tla_jar="$(resolve_runfile "$2")"
java_version_output="$("${java_bin}" -version 2>&1)" || {
  echo "Java runtime version check failed" >&2
  exit 1
}
grep -Eq ' version "17([."]|$)' <<<"${java_version_output}" || {
  echo "unexpected Java runtime: ${java_version_output}" >&2
  exit 1
}
printf 'DEVICE_MESH_REFINE_TOOLCHAIN checker=tlc release=1.7.4 java_major=17\n'

work_dir="${TEST_TMPDIR:?TEST_TMPDIR is required}/tlc-device-mesh-refine"

modules=(DeviceMeshModel.tla DeviceMeshFacts.tla MeshTopology.tla DeviceMeshRefine.tla)

stage() {
  local name="$1"
  shift
  mkdir -p "${work_dir}/${name}"
  local file
  for file in "${modules[@]}" "$@"; do
    cp "${fixture_dir}/${file}" "${work_dir}/${name}/${file}"
  done
}

# One worker everywhere, deliberately: the reported state counts are then
# reproducible, and a negative's search order does not depend on how many
# cores the machine happens to have. See the -Xss note in the header.
run_tlc() {
  local name="$1"
  local config="$2"
  local status=0
  (
    cd "${work_dir}/${name}"
    timeout 600 "${java_bin}" -XX:+UseParallelGC -Xss32m -cp "${tla_jar}" \
      tlc2.TLC -workers 1 -metadir "${work_dir}/${name}/states" \
      DeviceMeshRefine.tla -config "${config}.cfg"
  ) >"${work_dir}/${name}.log" 2>&1 || status=$?
  return "${status}"
}

distinct_states() {
  sed -n 's/.*[0-9]\+ states generated, \([0-9]\+\) distinct states found.*/\1/p' \
    "${work_dir}/$1.log" | tail -1
}

search_depth() {
  sed -n 's/^The depth of the complete state graph search is \([0-9]\+\)\..*/\1/p' \
    "${work_dir}/$1.log" | tail -1
}

max_outdegree() {
  sed -n 's/.*the maximum \([0-9]\+\) and the 95th percentile.*/\1/p' \
    "${work_dir}/$1.log" | tail -1
}

# Scalar constant as the cfg actually binds it, so a reported bound cannot
# drift from the bound that was checked.
cfg_scalar() {
  sed -n "s/^[[:space:]]*$2[[:space:]]*=[[:space:]]*\([A-Za-z0-9_]\+\).*/\1/p" \
    "${fixture_dir}/$1" | tail -1
}

cfg_body() {
  grep -v '^[[:space:]]*\\\*' "$1" | grep -v '^[[:space:]]*$'
}

# 0. Parse cost, measured rather than assumed. SANY does the parse and the
# semantic processing and stops, so this is the fixed cost every stage below
# pays before it generates a state.
mkdir -p "${work_dir}/parse"
for file in "${modules[@]}"; do
  cp "${fixture_dir}/${file}" "${work_dir}/parse/${file}"
done
parse_start="$(date +%s%N)"
parse_status=0
(
  cd "${work_dir}/parse"
  timeout 600 "${java_bin}" -Xss32m -cp "${tla_jar}" tla2sany.SANY \
    DeviceMeshRefine.tla
) >"${work_dir}/parse.log" 2>&1 || parse_status=$?
parse_end="$(date +%s%N)"
parse_output="$(<"${work_dir}/parse.log")"
[[ "${parse_status}" -eq 0 ]] \
  && ! formal_has_infrastructure_error "${parse_output}" || {
  cat "${work_dir}/parse.log" >&2
  echo "the refinement bridge does not parse" >&2
  exit 1
}
parse_ms=$(((parse_end - parse_start) / 1000000))
facts_bytes="$(wc -c <"${fixture_dir}/DeviceMeshFacts.tla")"
# Share of the parsed bytes that are the structural per-parameter placement
# facts, so a growing placement export shows up against the parse time it is
# charged to instead of disappearing into the total.
placement_bytes="$(
  awk '/BEGIN structural placement facts/,/END structural placement facts/' \
    "${fixture_dir}/DeviceMeshFacts.tla" | wc -c
)"
printf 'DEVICE_MESH_REFINE_PARSE result=success facts_bytes=%s placement_bytes=%s parse_ms=%s\n' \
  "${facts_bytes}" "${placement_bytes}" "${parse_ms}"

# 1. The bridge's inputs. A hand-written communicator table that disagreed
# with CommMembers would invalidate every claim below, so the mapping is
# derived from the recorded member sets and operations and then asserted. The
# derivation is printed by the module and echoed here, because a mapping
# quoted in prose is not the mapping the checker used.
stage mapping DeviceMeshRefineMapping.cfg
mapping_status=0
run_tlc mapping DeviceMeshRefineMapping || mapping_status=$?
mapping_output="$(<"${work_dir}/mapping.log")"
formal_classify_tlc_valid "${mapping_status}" "${mapping_output}" || {
  cat "${work_dir}/mapping.log" >&2
  echo "the observed-to-model communicator mapping or the derived issue" \
    "sequences did not satisfy the bridge's input assertions" >&2
  exit 1
}
grep -Fq 'DEVICE_MESH_REFINE_COMM_MAPPING' "${work_dir}/mapping.log" || {
  cat "${work_dir}/mapping.log" >&2
  echo "the checker did not print the derived communicator mapping" >&2
  exit 1
}
printf 'DEVICE_MESH_REFINE_MAPPING result=success exit=%s\n' "${mapping_status}"
sed -n '/DEVICE_MESH_REFINE_COMM_MAPPING/,/>>/p' "${work_dir}/mapping.log"

# 2. The refinement itself.
stage refine DeviceMeshRefine.cfg
refine_status=0
run_tlc refine DeviceMeshRefine || refine_status=$?
refine_output="$(<"${work_dir}/refine.log")"
formal_classify_tlc_transition_negative \
  "${refine_status}" "${refine_output}" "ReplayedRunIsNotAdmitted" || {
  cat "${work_dir}/refine.log" >&2
  echo "the observed issue order was not admitted by the DPxTP model" >&2
  exit 1
}
refine_issues="$(cfg_scalar DeviceMeshRefine.cfg MaxIssues)"
refine_skew="$(cfg_scalar DeviceMeshRefine.cfg MaxReplaySkew)"
[[ -n "${refine_issues}" && -n "${refine_skew}" ]] || {
  echo "could not read the replay bound out of DeviceMeshRefine.cfg" >&2
  exit 1
}
# The depth is the length of the witness path: the number of model steps that
# replay the trace. Reported so a reader can see the cost is linear in the
# trace and not a combinatorial search.
printf 'DEVICE_MESH_REFINE result=admitted witness=ReplayedRunIsNotAdmitted issues_per_rank=%s search_skew_bound=%s distinct_states=%s witness_depth=%s exit=%s\n' \
  "${refine_issues}" "${refine_skew}" "$(distinct_states refine)" \
  "$(search_depth refine)" "${refine_status}"

# 2b. The order-sensitivity pair. The positive above is a statement about the
# 108 cross-rank COLUMNS -- typable, agreeing, and drainable -- not about the
# absolute sequence, because no guard in DeviceMeshModel mentions which collective
# a position ought to carry. Rather than leave that as a caveat, both halves
# are checked: a permutation applied uniformly to all four ranks must be
# ADMITTED, and the same permutation applied to one rank must be REFUSED. The
# refused half is the evidence that the bridge detects the property NCCL
# actually imposes.
stage order_uniform DeviceMeshRefineUniformPermutation.cfg
order_uniform_status=0
run_tlc order_uniform DeviceMeshRefineUniformPermutation || order_uniform_status=$?
order_uniform_output="$(<"${work_dir}/order_uniform.log")"
formal_classify_tlc_transition_negative \
  "${order_uniform_status}" "${order_uniform_output}" \
  "ReplayedRunIsNotAdmitted" || {
  cat "${work_dir}/order_uniform.log" >&2
  echo "a permutation applied uniformly to every rank was NOT admitted, so" \
    "the bridge is refusing something the protocol permits: the columns are" \
    "unchanged and only their order is" >&2
  exit 1
}
printf 'DEVICE_MESH_REFINE_ORDER_UNIFORM permutation=reverse_every_rank result=admitted distinct_states=%s witness_depth=%s exit=%s\n' \
  "$(distinct_states order_uniform)" "$(search_depth order_uniform)" \
  "${order_uniform_status}"

stage order_single_rank DeviceMeshRefineSingleRankPermutation.cfg
order_single_status=0
run_tlc order_single_rank DeviceMeshRefineSingleRankPermutation \
  || order_single_status=$?
order_single_output="$(<"${work_dir}/order_single_rank.log")"
formal_classify_tlc_valid \
  "${order_single_status}" "${order_single_output}" || {
  cat "${work_dir}/order_single_rank.log" >&2
  echo "reversing one rank's issue order was not refused, so the bridge does" \
    "not detect a cross-rank agreement violation -- which is the one" \
    "property NCCL imposes and the one a real job hangs on" >&2
  exit 1
}
printf 'DEVICE_MESH_REFINE_ORDER_SINGLE_RANK permutation=reverse_one_rank result=refused distinct_states=%s exit=%s\n' \
  "$(distinct_states order_single_rank)" "${order_single_status}"

# 3. Confluence over a declared fragment, machine-checked. Terminated
# self-loops on AllDone alone and the state graph is acyclic otherwise, so an
# exhaustive run with no dead end says every schedule in the fragment reaches
# completion. A pass also certifies that the two cost-control bounds are not
# themselves obstructive, since a bound that blocked all progress would show
# up here as a deadlock.
stage confluence DeviceMeshRefineSkew.cfg
confluence_status=0
run_tlc confluence DeviceMeshRefineSkew || confluence_status=$?
confluence_output="$(<"${work_dir}/confluence.log")"
formal_classify_tlc_valid \
  "${confluence_status}" "${confluence_output}" || {
  cat "${work_dir}/confluence.log" >&2
  echo "the bounded-skew replay did not complete cleanly: some schedule of" \
    "the observed issue order within the bound fails to reach completion," \
    "or a cost-control bound blocks progress" >&2
  exit 1
}
confluence_issues="$(cfg_scalar DeviceMeshRefineSkew.cfg MaxIssues)"
confluence_skew="$(cfg_scalar DeviceMeshRefineSkew.cfg MaxReplaySkew)"
confluence_flight="$(cfg_scalar DeviceMeshRefineSkew.cfg MaxOutstanding)"
[[ -n "${confluence_issues}" && -n "${confluence_skew}" \
  && -n "${confluence_flight}" ]] || {
  echo "could not read the confluence bounds out of DeviceMeshRefineSkew.cfg" >&2
  exit 1
}
grep -Fq 'CHECK_DEADLOCK TRUE' "${fixture_dir}/DeviceMeshRefineSkew.cfg" || {
  echo "the confluence configuration does not enable deadlock detection, so" \
    "a schedule that cannot finish would go unreported" >&2
  exit 1
}
# The bounds belong in the token. A reader who meets only "no dead end" has no
# way to tell that this covers the first issues of each rank under a bounded
# spread and a bounded overlap window, not the whole 108-issue trace.
printf 'DEVICE_MESH_REFINE_CONFLUENCE result=no_dead_end bound_issues_per_rank=%s bound_issue_skew=%s bound_outstanding_per_comm=%s distinct_states=%s max_outdegree=%s exit=%s\n' \
  "${confluence_issues}" "${confluence_skew}" "${confluence_flight}" \
  "$(distinct_states confluence)" "$(max_outdegree confluence)" \
  "${confluence_status}"

# 4. And that fragment is not a serialized space. Without this the confluence
# result could rest on a space in which collectives never overlap, and a
# serialized schedule satisfies every guard trivially -- which is exactly the
# fiction the observed started/completed sub-order would have asserted.
stage overlap DeviceMeshRefineOverlap.cfg
overlap_diff="$(
  diff <(cfg_body "${fixture_dir}/DeviceMeshRefineSkew.cfg") \
       <(cfg_body "${fixture_dir}/DeviceMeshRefineOverlap.cfg") \
    || true
)"
# Sorted, because diff emits its hunks in file order and the two changed
# regions are not adjacent; sorting keeps the check sensitive to an extra or a
# missing line without pinning where in the file each one sits.
expected_overlap_diff="$(printf '%s\n' \
  '< CHECK_DEADLOCK TRUE' \
  '< INVARIANTS' \
  '< IssuedIsAReplayPrefix' \
  '< ReplayOutstandingIsWithinBound' \
  '< ReplaySkewIsWithinBound' \
  '< TypeOK' \
  '> CHECK_DEADLOCK FALSE' \
  '> INVARIANT NoTwoCollectivesRunConcurrently' | sort)"
overlap_diff_lines="$(
  grep -E '^[<>]' <<<"${overlap_diff}" | sed 's/^\([<>]\) */\1 /' | sort
)"
[[ "${overlap_diff_lines}" == "${expected_overlap_diff}" ]] || {
  printf 'overlap cfg differs from the confluence cfg in more than the\ninvariant and the deadlock switch:\n%s\n' \
    "${overlap_diff}" >&2
  exit 1
}
overlap_status=0
run_tlc overlap DeviceMeshRefineOverlap || overlap_status=$?
overlap_output="$(<"${work_dir}/overlap.log")"
formal_classify_tlc_transition_negative \
  "${overlap_status}" "${overlap_output}" \
  "NoTwoCollectivesRunConcurrently" || {
  cat "${work_dir}/overlap.log" >&2
  echo "the bounded-skew space contains no state with two collectives in" \
    "flight, so its confluence result covers serialized schedules only" >&2
  exit 1
}
printf 'DEVICE_MESH_REFINE_CONFLUENCE_OVERLAP invariant=NoTwoCollectivesRunConcurrently result=named_violation exit=%s\n' \
  "${overlap_status}"

# 5. The negative control, and where it is refused.
stage negative DeviceMeshRefineBad.cfg
negative_status=0
run_tlc negative DeviceMeshRefineBad || negative_status=$?
negative_output="$(<"${work_dir}/negative.log")"
formal_classify_tlc_valid "${negative_status}" "${negative_output}" || {
  cat "${work_dir}/negative.log" >&2
  echo "the transposed issue order was not refused by the DPxTP model" >&2
  exit 1
}

stage negative_reach DeviceMeshRefineBadReach.cfg
negative_reach_status=0
run_tlc negative_reach DeviceMeshRefineBadReach || negative_reach_status=$?
negative_reach_output="$(<"${work_dir}/negative_reach.log")"
formal_classify_tlc_transition_negative \
  "${negative_reach_status}" "${negative_reach_output}" \
  "RejectionHappensBeforeTheRendezvous" || {
  cat "${work_dir}/negative_reach.log" >&2
  echo "the transposed issue order was refused before reaching the" \
    "corrupted rendezvous, so the control exercises the wrong guard" >&2
  exit 1
}
printf 'DEVICE_MESH_REFINE_NEGATIVE result=rejected_at_rendezvous_guard distinct_states=%s reach_distinct_states=%s reject_exit=%s reach_exit=%s\n' \
  "$(distinct_states negative)" "$(distinct_states negative_reach)" \
  "${negative_status}" "${negative_reach_status}"

# 6. Guard isolation, proved directly. The same corrupted issue order, the
# same module, the same mutation -- only RequireMatchedIssueOrder differs. If
# relaxing that one constant makes the corruption complete, then that constant
# is what refused it in check 5, which no combination of "where did it stop"
# observations can establish on its own.
stage negative_relaxed DeviceMeshRefineBadRelaxed.cfg
relaxed_diff="$(
  diff <(cfg_body "${fixture_dir}/DeviceMeshRefineBad.cfg") \
       <(cfg_body "${fixture_dir}/DeviceMeshRefineBadRelaxed.cfg") \
    || true
)"
# Sorted, for the same reason as the overlap pair above.
expected_relaxed_diff="$(printf '%s\n' \
  '< MismatchedCollectiveNeverRuns' \
  '< RequireMatchedIssueOrder = TRUE' \
  '> RequireMatchedIssueOrder = FALSE' | sort)"
relaxed_diff_lines="$(
  grep -E '^[<>]' <<<"${relaxed_diff}" | sed 's/^\([<>]\) */\1 /' | sort
)"
[[ "${relaxed_diff_lines}" == "${expected_relaxed_diff}" ]] || {
  printf 'relaxed cfg differs from the negative cfg in more than the guard\nand the one invariant that only applies while it is on:\n%s\n' \
    "${relaxed_diff}" >&2
  exit 1
}
relaxed_status=0
run_tlc negative_relaxed DeviceMeshRefineBadRelaxed || relaxed_status=$?
relaxed_output="$(<"${work_dir}/negative_relaxed.log")"
formal_classify_tlc_transition_negative \
  "${relaxed_status}" "${relaxed_output}" "ReplayedRunIsNotAdmitted" || {
  cat "${work_dir}/negative_relaxed.log" >&2
  echo "relaxing RequireMatchedIssueOrder did not make the transposed issue" \
    "order complete, so the refusal in check 5 is not attributable to that" \
    "guard" >&2
  exit 1
}
printf 'DEVICE_MESH_REFINE_GUARD_ISOLATION relaxed=RequireMatchedIssueOrder result=admitted witness=ReplayedRunIsNotAdmitted exit=%s\n' \
  "${relaxed_status}"

# 7. The same corruption under the POSITIVE's guard set. Checks 5 and 6 both
# relax RequireUniformProgramOps, so neither runs with the guards the positive
# result was obtained under, and a reader could reasonably ask whether the
# corruption survives them. It does not: it is refused, and refused before the
# corrupted column can form -- no peer reaches that position -- which names the
# SPMD-program guard as the refuser there. Both refusals are correct -- a single-rank transposition is an
# agreement violation and both guards are entitled to catch it -- and this
# stage is what keeps the attribution in checks 5 and 6 from being read as the
# only refusal available.
stage negative_guarded DeviceMeshRefineBadUniform.cfg
guarded_diff="$(
  diff <(cfg_body "${fixture_dir}/DeviceMeshRefineBad.cfg") \
       <(cfg_body "${fixture_dir}/DeviceMeshRefineBadUniform.cfg") \
    || true
)"
expected_guarded_diff="$(printf '%s\n' \
  '< MismatchedCollectiveNeverRuns' \
  '< RequireUniformProgramOps = FALSE' \
  '> CorruptedColumnIsNeverFormed' \
  '> RequireUniformProgramOps = TRUE' | sort)"
guarded_diff_lines="$(
  grep -E '^[<>]' <<<"${guarded_diff}" | sed 's/^\([<>]\) */\1 /' | sort
)"
[[ "${guarded_diff_lines}" == "${expected_guarded_diff}" ]] || {
  printf 'guard-set cfg differs from the negative cfg in more than the SPMD\nguard and the one invariant that only applies while it is off:\n%s\n' \
    "${guarded_diff}" >&2
  exit 1
}
guarded_status=0
run_tlc negative_guarded DeviceMeshRefineBadUniform || guarded_status=$?
guarded_output="$(<"${work_dir}/negative_guarded.log")"
formal_classify_tlc_valid "${guarded_status}" "${guarded_output}" || {
  cat "${work_dir}/negative_guarded.log" >&2
  echo "under the positive's guard set the transposed issue order was not" \
    "refused before the corrupted column formed, so the SPMD-program guard" \
    "is not the refuser there" >&2
  exit 1
}
printf 'DEVICE_MESH_REFINE_NEGATIVE_GUARD_SET guards=positive result=refused_at_spmd_guard witness=CorruptedColumnIsNeverFormed distinct_states=%s exit=%s\n' \
  "$(distinct_states negative_guarded)" "${guarded_status}"
