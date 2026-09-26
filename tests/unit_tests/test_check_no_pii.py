# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from scripts import check_no_pii


def _scan_text(tmp_path: Path, text: str) -> list[check_no_pii.Finding]:
    path = tmp_path / "fixture.txt"
    path.write_text(text)
    return check_no_pii.scan_file(str(path))


def test_official_bcr_metadata_versions_are_not_ipv4(tmp_path: Path) -> None:
    findings = _scan_text(
        tmp_path,
        "\n".join(
            [
                '"https://bcr.bazel.build/modules/'
                'swift_argument_parser/1.3.1.1/MODULE.bazel": "digest",',  # pii-allow
                '"https://bcr.bazel.build/modules/'
                'swift_argument_parser/1.3.1.2/source.json": "digest",',  # pii-allow
            ]
        ),
    )

    assert findings == []


def test_bcr_context_does_not_hide_ipv4_outside_version_segment(
    tmp_path: Path,
) -> None:
    findings = _scan_text(
        tmp_path,
        "\n".join(
            [
                "peer=1.3.1.1",  # pii-allow
                '"https://bcr.bazel.build/modules/rules_shell/0.6.1/'
                'MODULE.bazel": "mirror 1.3.1.2",',  # pii-allow
                '"https://bcr.bazel.build/modules/1.3.1.2/0.6.1/'  # pii-allow
                'MODULE.bazel": "digest",',  # pii-allow
            ]
        ),
    )

    assert [
        (finding.lineno, finding.category, finding.preview) for finding in findings
    ] == [
        (1, "routable-ipv4", "1.3.1.1"),  # pii-allow
        (2, "routable-ipv4", "1.3.1.2"),  # pii-allow
        (3, "routable-ipv4", "1.3.1.2"),  # pii-allow
    ]
