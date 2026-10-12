# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Behavior tests for Falcon native evidence and legacy normalization."""

from __future__ import annotations

import hashlib

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parents[2]


def _native_attempt() -> dict:
    return {
        "run_id": "falcon-step4-lm-A5-seed0",
        "attempt_id": "falcon-a5-seed0-attempt0",
        "lane": "science",
        "mode": "train",
        "arm": "A5",
        "claim_label": "representative_training",
        "evidence_tier": "tier0",
        "environment_class": "rootfs_b200",
        "logical_run": {"workload": "step4-lm", "arm": "A5", "seed": 0},
        "processes": [
            {
                "process_id": "trainer.rank0",
                "role": "trainer",
                "global_rank": 0,
                "local_rank": 0,
                "world_size": 1,
                "host_name": "b200-0",
            }
        ],
        "mesh": {"axes": {"dp": 1, "tp": 1}},
        "device": {"type": "cuda", "index": 0, "uuid": "GPU-test"},
        "clocks": {
            "started_utc": "2026-09-08T00:00:00Z",
            "ended_utc": "2026-09-08T00:01:00Z",
        },
        "steps": {"requested": 10, "completed": 10},
        "phases": [{"name": "train", "outcome": "completed"}],
        "data_lineage": {"dataset_id": "fineweb-fixed"},
        "checkpoint_lineage": {
            "input_checkpoint": None,
            "output_checkpoint": "step-10",
        },
        "artifacts": [{"path": "metrics.jsonl", "kind": "metrics"}],
        "outcome": {"status": "completed", "reason": None},
    }


def test_native_attempt_bundle_rejects_missing_identity_and_is_immutable(
    tmp_path: Path,
):
    """Deleting required rank identity must make the Falcon bundle invalid."""
    from torchtitan.experiments.falcon.evidence import (
        EvidenceConflictError,
        EvidenceContractError,
        write_native_attempt_bundle,
    )

    attempt = _native_attempt()
    del attempt["processes"][0]["global_rank"]
    with pytest.raises(EvidenceContractError, match="global_rank"):
        write_native_attempt_bundle(tmp_path, attempt)

    attempt = _native_attempt()
    del attempt["logical_run"]
    with pytest.raises(EvidenceContractError, match="logical_run"):
        write_native_attempt_bundle(tmp_path, attempt)

    attempt = _native_attempt()
    bundle_path = write_native_attempt_bundle(tmp_path, attempt)
    assert (
        json.loads((bundle_path / "manifest.json").read_text())["run_id"]
        == "falcon-step4-lm-A5-seed0"
    )
    assert (
        json.loads((bundle_path / "artifact_index.json").read_text())
        == attempt["artifacts"]
    )
    assert json.loads((bundle_path / "outcome.json").read_text()) == attempt["outcome"]

    changed = _native_attempt()
    changed["outcome"] = {
        "status": "failed",
        "reason": "not the same immutable attempt",
    }
    with pytest.raises(EvidenceConflictError, match="immutable"):
        write_native_attempt_bundle(tmp_path, changed)


@pytest.mark.parametrize("bad_id", ("../escape", "/tmp/escape", "..", "a/b", "a\\b"))
def test_native_bundle_rejects_path_escape_ids(tmp_path: Path, bad_id: str):
    """Path-like run or attempt IDs must never escape the evidence root."""
    from torchtitan.experiments.falcon.evidence import (
        EvidenceContractError,
        write_native_attempt_bundle,
    )

    attempt = _native_attempt()
    attempt["attempt_id"] = bad_id
    with pytest.raises(EvidenceContractError, match="safe path segment"):
        write_native_attempt_bundle(tmp_path, attempt)
    assert not list(tmp_path.iterdir())


def test_native_run_id_binds_one_workload_arm_seed_and_allows_restart(tmp_path: Path):
    """A matrix cell has one canonical run ID while attempts remain distinct."""
    from torchtitan.experiments.falcon.evidence import (
        EvidenceContractError,
        write_native_attempt_bundle,
    )

    first = _native_attempt()
    first_path = write_native_attempt_bundle(tmp_path, first)
    retry = _native_attempt()
    retry["attempt_id"] = "falcon-a5-seed0-attempt1"
    retry_path = write_native_attempt_bundle(tmp_path, retry)
    assert first_path.parent == retry_path.parent

    wrong_cell = _native_attempt()
    wrong_cell["arm"] = "A4"
    wrong_cell["logical_run"] = {"workload": "step4-lm", "arm": "A4", "seed": 0}
    with pytest.raises(EvidenceContractError, match="canonical"):
        write_native_attempt_bundle(tmp_path, wrong_cell)


def test_native_bundle_rejects_invalid_nested_contract_and_recovers_from_staging_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Invalid types and a failed staged publication cannot create partial bundles."""
    import torchtitan.experiments.falcon.evidence as evidence

    malformed = _native_attempt()
    malformed["processes"][0]["global_rank"] = True
    with pytest.raises(evidence.EvidenceContractError, match="integer"):
        evidence.write_native_attempt_bundle(tmp_path, malformed)

    malformed = _native_attempt()
    malformed["mesh"]["axes"]["dp"] = 2
    with pytest.raises(evidence.EvidenceContractError, match="mesh size"):
        evidence.write_native_attempt_bundle(tmp_path, malformed)

    malformed = _native_attempt()
    malformed["clocks"]["started_utc"] = "2026-09-08T00:00:00"
    with pytest.raises(evidence.EvidenceContractError, match="timezone"):
        evidence.write_native_attempt_bundle(tmp_path, malformed)

    writes = 0
    original = evidence._write_bundle_file

    def fail_second_write(path: Path, contents: bytes) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("injected staging failure")
        original(path, contents)

    monkeypatch.setattr(evidence, "_write_bundle_file", fail_second_write)
    with pytest.raises(OSError, match="injected"):
        evidence.write_native_attempt_bundle(tmp_path, _native_attempt())
    assert not (tmp_path / "runs" / "falcon-step4-lm-A5-seed0").exists()

    monkeypatch.setattr(evidence, "_write_bundle_file", original)
    assert evidence.write_native_attempt_bundle(tmp_path, _native_attempt()).is_dir()


@pytest.mark.parametrize(
    ("mutate", "error"),
    (
        (lambda attempt: attempt["device"].update({"index": True}), "integer"),
        (lambda attempt: attempt["steps"].update({"completed": 11}), "must not exceed"),
        (
            lambda attempt: attempt.update({"data_lineage": {"dataset_id": None}}),
            "lineage",
        ),
        (lambda attempt: attempt.update({"outcome": {"status": ""}}), "outcome.status"),
        (
            lambda attempt: attempt["phases"].append(
                {"name": "train", "outcome": "done"}
            ),
            "phase names",
        ),
        (
            lambda attempt: attempt["artifacts"][0].update({"path": "../escape"}),
            "repository-relative",
        ),
    ),
)
def test_native_bundle_rejects_each_nested_contract_class(
    tmp_path: Path, mutate, error: str
):
    """Every nested native evidence class has behavioral validation coverage."""
    from torchtitan.experiments.falcon.evidence import (
        EvidenceContractError,
        write_native_attempt_bundle,
    )

    attempt = _native_attempt()
    mutate(attempt)
    with pytest.raises(EvidenceContractError, match=error):
        write_native_attempt_bundle(tmp_path, attempt)


def test_legacy_import_is_idempotent_resolves_sources_and_rejects_corruption(
    tmp_path: Path,
):
    """Changing a hashed source must not silently rewrite a legacy ledger."""
    from torchtitan.experiments.falcon.evidence import (
        EvidenceConflictError,
        import_legacy_fixtures,
        LegacyFixture,
        verify_legacy_ledger,
    )

    source = tmp_path / "legacy.json"
    source.write_text('{"outcome":"complete"}\n')
    ledger = tmp_path / "legacy-ledger.json"
    fixtures = [
        LegacyFixture(
            key="one-completed-fixture",
            source_path="legacy.json",
            artifact_role="raw_outcome",
            record_status="valid",
            facts={"arm": "A5", "seed": 0},
            unavailable={"run_id": "legacy compact JSON has no declared run identity"},
            source_selector="summary",
        )
    ]

    first = import_legacy_fixtures(tmp_path, ledger, fixtures)
    original = first.read_bytes()
    assert import_legacy_fixtures(tmp_path, ledger, fixtures).read_bytes() == original
    assert verify_legacy_ledger(tmp_path, ledger) == 1

    source.write_text('{"outcome":"corrupted"}\n')
    with pytest.raises(EvidenceConflictError, match="source digest changed"):
        import_legacy_fixtures(tmp_path, ledger, fixtures)


def test_legacy_import_collapses_equal_semantic_sources_and_rejects_conflicts(
    tmp_path: Path,
):
    """Derived duplicates collapse, but competing evidence for one cell cannot."""
    from torchtitan.experiments.falcon.evidence import (
        EvidenceConflictError,
        import_legacy_fixtures,
        LegacyFixture,
    )

    for name in ("raw.json", "derived.json", "conflict.json"):
        (tmp_path / name).write_text(f'{{"source":"{name}"}}\n')
    shared = {
        "campaign_step": "step4",
        "mode": "train",
        "arm": "A5",
        "seed": 0,
        "requested_steps": 8000,
        "metric": 4.7,
    }
    equal = [
        LegacyFixture("raw", "raw.json", "outcome", "valid", shared, {}, "summary"),
        LegacyFixture(
            "derived", "derived.json", "outcome", "valid", shared, {}, "summary"
        ),
    ]
    payload = json.loads(
        import_legacy_fixtures(tmp_path, tmp_path / "equal.json", equal).read_text()
    )
    assert len(payload["records"]) == 1

    conflicting = [
        equal[0],
        LegacyFixture(
            "conflict",
            "conflict.json",
            "outcome",
            "valid",
            {**shared, "metric": 4.8},
            {},
            "summary",
        ),
    ]
    with pytest.raises(EvidenceConflictError, match="semantic evidence conflict"):
        import_legacy_fixtures(tmp_path, tmp_path / "conflict-ledger.json", conflicting)


def test_known_legacy_falcon_inventory_is_normalized_without_derived_duplicates(
    tmp_path: Path,
):
    """A duplicate derived table must not become another logical training run."""
    from torchtitan.experiments.falcon.evidence import (
        import_known_legacy_falcon_evidence,
        verify_known_pre_b09_ledger,
    )

    ledger = import_known_legacy_falcon_evidence(REPO_ROOT, tmp_path / "ledger.json")
    payload = json.loads(ledger.read_text())
    records = payload["records"]

    assert len(records) == 55
    assert sum(record["record_status"] == "valid" for record in records) == 20
    assert sum(record["record_status"] == "smoke" for record in records) == 23
    assert (
        sum(record["record_status"] == "learnability_smoke" for record in records) == 3
    )
    assert sum(record["record_status"] == "incomplete" for record in records) == 2
    assert sum(record["record_status"] == "omitted" for record in records) == 5
    assert {record["record_status"] for record in records} >= {
        "valid",
        "smoke",
        "invalid",
        "incomplete",
        "omitted",
    }
    assert len({record["import_id"] for record in records}) == len(records)
    assert all(record["source"]["digest"] for record in records)
    expected_source_digests = {
        source["path"]: hashlib.sha256(
            (REPO_ROOT / source["path"]).read_bytes()
        ).hexdigest()
        for source in payload["sources"]
    }
    assert {
        source["path"]: source["digest"] for source in payload["sources"]
    } == expected_source_digests
    assert {record["import_id"] for record in records} == {
        "legacy-"
        + hashlib.sha256(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_path": record["source"]["path"],
                    "source_digest": record["source"]["digest"],
                    "source_selector": record["source_selector"],
                    "artifact_role": record["artifact_role"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n"
        ).hexdigest()[:24]
        for record in records
    }
    assert payload["inventory_subtotals"]["campaign_b_recap"]["record_count"] == 52
    classifications = {
        artifact["classification"]: set() for artifact in payload["artifact_index"]
    }
    for artifact in payload["artifact_index"]:
        classifications[artifact["classification"]].add(artifact["path"])
    assert len(classifications["associated_checkpoint"]) == 12
    assert len(classifications["associated_hero_status"]) == 6
    assert len(classifications["associated_hero_log"]) == 3
    assert len(classifications["copied_artifact"]) == 12
    assert (
        "experiments/falcon/results/ablations/combined/table.json"
        in classifications["derived_artifact"]
    )
    assert all(record["source_selector"] for record in records)
    assert not any("/combined/" in record["source"]["path"] for record in records)
    assert not any(record["source"]["path"].endswith("table.md") for record in records)
    addition_records = [
        record
        for record in records
        if record["artifact_role"] == "raw_ablation_table"
        and record["inferred_linkage"]
        and record["inferred_linkage"].get("kind") == "addition"
    ]
    assert addition_records
    assert all("id_acc" not in record["facts"] for record in addition_records)
    assert all(
        "training_bank_exact_suffix_accuracy" in record["facts"]
        for record in addition_records
    )
    assert all(
        record["source"]["path"]
        != "experiments/falcon/results/ablations/combined/table.json"
        for record in records
    )
    assert verify_known_pre_b09_ledger(REPO_ROOT, ledger) == len(records)


def test_known_inventory_preserves_exact_identity_outcomes_claims_and_artifacts(
    tmp_path: Path,
):
    """Campaign B keeps exact science identities without promoting smoke evidence."""
    from torchtitan.experiments.falcon.evidence import (
        import_known_legacy_falcon_evidence,
    )

    payload = json.loads(
        import_known_legacy_falcon_evidence(
            REPO_ROOT, tmp_path / "ledger.json"
        ).read_text()
    )
    records = payload["records"]
    full_table = "experiments/falcon/results/ablations/full_2seed_8000/table.json"
    a1_table = "experiments/falcon/results/ablations/full_A1_2seed_8000/table.json"
    expected_valid = (
        {
            (full_table, f"arms[{arm},{seed}]")
            for arm in ("A0", "A2", "A4", "A5")
            for seed in (0, 1)
        }
        | {
            (full_table, f"addition[{arm}_seed{seed}]")
            for arm in ("A0", "A2", "A4", "A5")
            for seed in (0, 1)
        }
        | {(a1_table, "arms[A1,0]")}
        | {
            (
                f"experiments/falcon/results/hero/hero_20k/{arm}_seed0/hero_outcome.json",
                "hero_outcome",
            )
            for arm in ("A0", "A1", "A5")
        }
    )
    assert {
        (record["source"]["path"], record["source_selector"])
        for record in records
        if record["record_status"] == "valid"
    } == expected_valid

    a3 = next(
        record
        for record in records
        if record["import_id"]
        and record["facts"].get("arm") == "A3"
        and record["record_status"] == "incomplete"
    )
    assert a3["facts"]["seed"] == a3["inferred_linkage"]["seed"] == 0
    v3 = [
        record
        for record in records
        if record["inferred_linkage"]
        and record["inferred_linkage"].get("workload") == "tiny_overfit_30k_profiler"
    ]
    assert len(v3) == 2
    assert v3[0]["inferred_linkage"] == v3[1]["inferred_linkage"]
    failed = next(record for record in v3 if record["record_status"] == "invalid")
    expected_failed_outcome = json.loads(
        (
            REPO_ROOT
            / "experiments/falcon/results/tooling/tiny_inject_v3/run_evidence/"
            "f87875d8-73a1-4a71-be68-6a575d008bd4/"
            "5fc6fa7c-8b1f-4d17-9677-20b95b1830fb/processes/"
            "trainer.core.global_rank_000000/outcome.json"
        ).read_text()
    )
    assert failed["facts"]["source_process_outcome"] == expected_failed_outcome
    assert failed["facts"]["processes"] == [expected_failed_outcome]
    assert failed["facts"]["outcome"] == "failed"
    assert "processes" not in failed["unavailable"]
    assert "outcome" not in failed["unavailable"]

    additions = [
        record
        for record in records
        if record["inferred_linkage"]
        and record["inferred_linkage"].get("kind") == "addition"
    ]
    assert all(
        record["facts"]["claim_label"] == "smoke"
        for record in additions
        if "/dry_run/" in record["source"]["path"]
    )
    assert all(
        record["facts"]["source_declared_claim_label"]
        == "representative_small + replicated_eval"
        and record["facts"]["claim_label"] == "representative_small"
        for record in additions
        if "/full_" in record["source"]["path"]
    )

    artifact_paths = {artifact["path"] for artifact in payload["artifact_index"]}
    raw_paths = {
        source["path"]
        for record in records
        for source in [record["source"], *record["related_sources"]]
    }
    expected_non_raw = {
        "experiments/falcon/results/tooling/tiny_inject_v3/console.log",
        "experiments/falcon/results/ablations/combined/table.json",
        "experiments/falcon/results/ablations/combined/table.md",
    }
    expected_non_raw.update(
        {
            f"experiments/falcon/results/ablations/{arm}/table.md"
            for arm in ("dry_run", "full_2seed_8000", "full_A1_2seed_8000")
        }
    )
    expected_non_raw.update(
        {
            ".scratch/falcon-fast-weight-attention/evidence/ablations/"
            f"{arm}/{suffix}"
            for arm in ("combined", "dry_run", "full_2seed_8000", "full_A1_2seed_8000")
            for suffix in ("table.json", "table.md")
        }
    )
    expected_non_raw.update(
        {
            ".scratch/falcon-fast-weight-attention/evidence/ebpf-nosudo-REPORT.md",
            ".scratch/falcon-fast-weight-attention/evidence/rootfs-all-tools-REPORT.md",
            ".scratch/falcon-fast-weight-attention/evidence/rootfs-privileged-REPORT.md",
            ".scratch/falcon-fast-weight-attention/evidence/tooling-REPORT.md",
        }
    )
    for tier, steps in (("hero_20k", (19000, 19500, 20000)), ("hero_dry", (4,))):
        for arm in ("A0", "A1", "A5"):
            base = f"experiments/falcon/results/hero/{tier}/{arm}_seed0"
            expected_non_raw.add(f"{base}/hero_status.json")
            expected_non_raw.update(f"{base}/checkpoint/step-{step}" for step in steps)
            if tier == "hero_20k":
                expected_non_raw.update({f"{base}/hero.host.pid", f"{base}/hero.log"})
    expected_non_raw.update(
        {
            "experiments/falcon/results/ebpf_nosudo_20260906T050950Z",
            "experiments/falcon/results/tooling/attach_v1",
            "experiments/falcon/results/tooling/attach_v3",
            "experiments/falcon/results/tooling/nsys_diag",
            "experiments/falcon/results/tooling/py_profilers",
            "experiments/falcon/results/tooling/rootfs_all_tools",
            "experiments/falcon/results/tooling/rootfs_privileged",
            "experiments/falcon/results/tooling/attach_privileged",
        }
    )
    assert artifact_paths == raw_paths | expected_non_raw


def test_legacy_verifier_rejects_tampered_json_selector_and_subtotals(tmp_path: Path):
    """Ledger verification checks JSON integrity, selectors, and derived totals."""
    from torchtitan.experiments.falcon.evidence import (
        EvidenceConflictError,
        EvidenceContractError,
        import_known_legacy_falcon_evidence,
        verify_known_pre_b09_ledger,
    )

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{")
    with pytest.raises(EvidenceContractError, match="valid JSON"):
        verify_known_pre_b09_ledger(REPO_ROOT, invalid_json)

    ledger = import_known_legacy_falcon_evidence(REPO_ROOT, tmp_path / "ledger.json")
    payload = json.loads(ledger.read_text())
    payload["inventory_subtotals"]["campaign_b_recap"]["status_counts"]["valid"] += 1
    ledger.write_text(json.dumps(payload))
    with pytest.raises(EvidenceConflictError, match="canonical catalog"):
        verify_known_pre_b09_ledger(REPO_ROOT, ledger)

    selector_ledger = import_known_legacy_falcon_evidence(
        REPO_ROOT, tmp_path / "selector-ledger.json"
    )
    payload = json.loads(selector_ledger.read_text())
    record = payload["records"][0]
    record["source_selector"] = "lines[999999:999999]"
    identity = {
        "schema_version": 1,
        "source_path": record["source"]["path"],
        "source_digest": record["source"]["digest"],
        "source_selector": record["source_selector"],
        "artifact_role": record["artifact_role"],
    }
    record["import_id"] = (
        "legacy-"
        + hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        ).hexdigest()[:24]
    )
    selector_ledger.write_text(json.dumps(payload))
    with pytest.raises(EvidenceConflictError, match="canonical catalog"):
        verify_known_pre_b09_ledger(REPO_ROOT, selector_ledger)


@pytest.mark.parametrize(
    ("mutation", "refresh_ledger_binding"),
    (
        (
            lambda payload: payload["records"][0]["facts"].update(
                {"metric": "tampered"}
            ),
            True,
        ),
        (
            lambda payload: payload["records"][0].update(
                {"inferred_linkage": {"workload": "tampered"}}
            ),
            True,
        ),
        (
            lambda payload: payload["records"][0]["unavailable"].update(
                {"outcome": "tampered"}
            ),
            True,
        ),
        (
            lambda payload: payload["records"][0]["source_declared_ids"].update(
                {"run_id": "tampered"}
            ),
            True,
        ),
        (
            lambda payload: payload["artifact_index"][0].update({"parent": "tampered"}),
            False,
        ),
    ),
)
def test_legacy_verifier_rejects_valid_json_payload_tampering(
    tmp_path: Path, mutation, refresh_ledger_binding: bool
):
    """A valid JSON edit cannot silently alter normalized evidence provenance."""
    from torchtitan.experiments.falcon.evidence import (
        EvidenceConflictError,
        import_known_legacy_falcon_evidence,
        verify_known_pre_b09_ledger,
    )

    ledger = import_known_legacy_falcon_evidence(REPO_ROOT, tmp_path / "ledger.json")
    payload = json.loads(ledger.read_text())
    mutation(payload)
    if refresh_ledger_binding:
        from torchtitan.experiments.falcon.evidence import _ledger_binding

        payload["ledger_binding"] = _ledger_binding(payload)
    ledger.write_text(json.dumps(payload))
    with pytest.raises(EvidenceConflictError, match="canonical catalog"):
        verify_known_pre_b09_ledger(REPO_ROOT, ledger)


def test_known_catalog_rejects_coordinated_binding_rewrite(tmp_path: Path):
    """The pre-B09 catalog is anchored by regenerated importer output, not hashes."""
    from torchtitan.experiments.falcon.evidence import (
        _ledger_binding,
        _source_binding,
        EvidenceContractError,
        import_known_legacy_falcon_evidence,
        verify_known_pre_b09_ledger,
    )

    ledger = import_known_legacy_falcon_evidence(REPO_ROOT, tmp_path / "ledger.json")
    payload = json.loads(ledger.read_text())
    record = payload["records"][0]
    record["facts"]["coordinated_edit"] = "tampered"
    record["inferred_linkage"] = {"workload": "coordinated-edit"}
    record["source_binding"] = _source_binding(
        REPO_ROOT,
        source_path=record["source"]["path"],
        source_selector=record["source_selector"],
        facts=record["facts"],
        inferred_linkage=record["inferred_linkage"],
        source_declared_ids=record["source_declared_ids"],
        unavailable=record["unavailable"],
    )
    payload["ledger_binding"] = _ledger_binding(payload)
    del payload["catalog_id"]
    ledger.write_text(json.dumps(payload))
    with pytest.raises(EvidenceContractError, match="catalog_id"):
        verify_known_pre_b09_ledger(REPO_ROOT, ledger)


@pytest.mark.parametrize(
    ("catalog_id", "remove_marker"),
    (
        pytest.param(None, True, id="missing"),
        pytest.param(None, False, id="null"),
        pytest.param("falcon_pre_b09_v2", False, id="wrong"),
        pytest.param("third_party_catalog", False, id="unknown"),
    ),
)
def test_known_catalog_verifier_rejects_untrusted_catalog_marker(
    tmp_path: Path, catalog_id: str | None, remove_marker: bool
):
    """Canonical verification fails closed when its stored marker is not exact."""
    from torchtitan.experiments.falcon.evidence import (
        EvidenceContractError,
        import_known_legacy_falcon_evidence,
        verify_known_pre_b09_ledger,
    )

    ledger = import_known_legacy_falcon_evidence(REPO_ROOT, tmp_path / "ledger.json")
    payload = json.loads(ledger.read_text())
    if remove_marker:
        del payload["catalog_id"]
    else:
        payload["catalog_id"] = catalog_id
    ledger.write_text(json.dumps(payload))

    with pytest.raises(EvidenceContractError, match="catalog_id"):
        verify_known_pre_b09_ledger(REPO_ROOT, ledger)


def test_known_and_generic_verifiers_enforce_distinct_trust_modes(tmp_path: Path):
    """Neither verifier may silently infer verification strength from ledger bytes."""
    from torchtitan.experiments.falcon.evidence import (
        EvidenceContractError,
        import_known_legacy_falcon_evidence,
        import_legacy_fixtures,
        LegacyFixture,
        verify_known_pre_b09_ledger,
        verify_legacy_ledger,
    )

    source = tmp_path / "legacy.json"
    source.write_text('{"outcome":"complete"}\n')
    ad_hoc_ledger = import_legacy_fixtures(
        tmp_path,
        tmp_path / "ad-hoc-ledger.json",
        [
            LegacyFixture(
                key="one-completed-fixture",
                source_path="legacy.json",
                artifact_role="raw_outcome",
                record_status="valid",
                facts={"arm": "A5", "seed": 0},
                unavailable={
                    "run_id": "legacy compact JSON has no declared run identity"
                },
                source_selector="summary",
            )
        ],
    )
    assert "catalog_id" not in json.loads(ad_hoc_ledger.read_text())
    assert verify_legacy_ledger(tmp_path, ad_hoc_ledger) == 1
    with pytest.raises(EvidenceContractError, match="catalog_id"):
        verify_known_pre_b09_ledger(tmp_path, ad_hoc_ledger)

    known_ledger = import_known_legacy_falcon_evidence(
        REPO_ROOT, tmp_path / "known-ledger.json"
    )
    with pytest.raises(EvidenceContractError, match="verify_known_pre_b09_ledger"):
        verify_legacy_ledger(REPO_ROOT, known_ledger)


def test_verify_ledger_cli_always_uses_known_catalog_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """An alternate output path cannot downgrade the CLI to generic verification."""
    import sys

    import torchtitan.experiments.falcon.evidence as evidence

    source = tmp_path / "legacy.json"
    source.write_text('{"outcome":"complete"}\n')
    ledger = evidence.import_legacy_fixtures(
        tmp_path,
        tmp_path / "ad-hoc-ledger.json",
        [
            evidence.LegacyFixture(
                key="one-completed-fixture",
                source_path="legacy.json",
                artifact_role="raw_outcome",
                record_status="valid",
                facts={"arm": "A5", "seed": 0},
                unavailable={
                    "run_id": "legacy compact JSON has no declared run identity"
                },
                source_selector="summary",
            )
        ],
    )
    monkeypatch.setenv("TORCHTITAN_IN_ROOTFS", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "falcon-evidence",
            "verify-ledger",
            "--repo-root",
            str(tmp_path),
            "--output",
            str(ledger),
        ],
    )

    with pytest.raises(evidence.EvidenceContractError, match="catalog_id"):
        evidence._main()
