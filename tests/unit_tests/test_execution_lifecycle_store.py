# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Wave F1 atomic writer and validator tests.

Terminal files are immutable; raw process records and artifact indexes are
append-only (roadmap 3.3). These tests exercise the atomic JSON writer, the
append-only JSONL writer, and the immutability guard. Host-testable.
"""

from __future__ import annotations

import json

import pytest

from torchtitan.experiments.execution import store


def test_write_terminal_json_is_atomic_and_readable(tmp_path):
    path = tmp_path / "outcome.json"
    store.write_terminal_json(path, {"kind": "outcome", "n": 1})
    assert json.loads(path.read_text()) == {"kind": "outcome", "n": 1}
    # No temp files left behind.
    assert list(tmp_path.glob(".*")) == []


def test_terminal_json_is_immutable(tmp_path):
    path = tmp_path / "outcome.json"
    store.write_terminal_json(path, {"n": 1})
    with pytest.raises(FileExistsError):
        store.write_terminal_json(path, {"n": 2})
    # Original content is preserved.
    assert json.loads(path.read_text()) == {"n": 1}


def test_append_jsonl_preserves_prior_rows(tmp_path):
    path = tmp_path / "events.jsonl"
    store.append_jsonl(path, {"seq": 0})
    store.append_jsonl(path, {"seq": 1})
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert [row["seq"] for row in rows] == [0, 1]


def test_read_jsonl_round_trips(tmp_path):
    path = tmp_path / "events.jsonl"
    store.append_jsonl(path, {"a": 1})
    store.append_jsonl(path, {"b": 2})
    assert store.read_jsonl(path) == [{"a": 1}, {"b": 2}]


def test_read_jsonl_missing_file_is_empty(tmp_path):
    assert store.read_jsonl(tmp_path / "nope.jsonl") == []
