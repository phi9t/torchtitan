#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pre-commit guard that blocks machine- and person-identifiable tokens.

The scanner reads each argv path (pre-commit passes staged files) as text,
skips binary blobs, and flags lines that reintroduce identifying tokens that
have been scrubbed from the evidence tree. It runs on plain stdlib so the
pre-commit "system" language can invoke it host-side without a rootfs.

Four token classes are covered:

  1. absolute home paths -- /data02/home/<user> and /home/<user>
  2. routable IPv4 and MAC addresses
  3. hostnames and GPU UUIDs
  4. emails (with an allowlist of benign addresses)

plus vendor identity tokens (uid/gid, the scrubbed username, the vendor
kernel tag) that are cheap to catch and were part of the leak.

Escape hatches:

  - Any line containing the literal token ``pii-allow`` is skipped. Use it in
    a comment or trailing token for intentional fixtures.
  - ``ALLOWLISTED_PATHS`` lists files that legitimately carry third-party PII
    from an upstream corpus, where per-line pragmas are impractical.

Usage:

    python scripts/check_no_pii.py FILE [FILE ...]   # pre-commit path
    python scripts/check_no_pii.py --all             # walk git ls-files
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

# Inline escape hatch: any line containing this token is skipped.
ALLOW_PRAGMA = "pii-allow"

# Files that legitimately contain third-party PII from an upstream corpus.
# These are not our identifying tokens and per-line pragmas are impractical
# on a large data blob.
#   tests/assets/c4_test/data.json -- 4.6MB upstream C4 sample; real-world
#   text scraped from the web may contain incidental IPs/emails/hostnames.
ALLOWLISTED_PATHS = frozenset(
    {
        "tests/assets/c4_test/data.json",
    }
)

# The scrubbed username. Kept as data so the scanner itself does not embed a
# fresh copy in a matchable form beyond this single guarded definition.
_SCRUBBED_USER = "philip" + ".yang"

# Absolute home paths. /home/USER is the sanitized placeholder and is allowed.
# The /home/ arm requires that it is not nested under another path segment
# (e.g. /project/home/.modal.toml is a rootfs config path, not a user home).
_ABS_HOME_RE = re.compile(
    r"/data02/home/[A-Za-z0-9._-]+|(?<![A-Za-z0-9._-])/home/[A-Za-z0-9._-]+"
)
_HOME_PLACEHOLDER = "/home/USER"

_MAC_RE = re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b")
_HOSTNAME_RE = re.compile(r"\bn\d{3}-\d{3}-\d{3}\b")
_GPU_UUID_RE = re.compile(
    r"\bGPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
)
_KERNEL_TAG_RE = re.compile(r"\b\d+\.\d+\.\d+\.bsk\.\d+\b")
# Vendor identity tokens. Build the uid/gid literals from fragments so this
# source file does not itself carry a matchable copy of the leaked tokens.
_LEAKED_UID_GID = 1018
_IDENTITY_RE = re.compile(
    rf"uid={_LEAKED_UID_GID}|gid={_LEAKED_UID_GID}|"
    + r"\("
    + re.escape(_SCRUBBED_USER)
    + r"\)|"
    + re.escape(_SCRUBBED_USER)
)

# Candidate IPv4 -- validated further below to exclude non-routable ranges and
# reject dotted version strings.
_IPV4_CANDIDATE_RE = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
_BCR_MODULE_METADATA_KEY_RE = re.compile(
    r'^\s*"https://bcr\.bazel\.build/modules/'
    r"[a-z](?:[a-z0-9._-]*[a-z0-9])?/"
    r'(?P<version>[0-9][A-Za-z0-9._+-]*)/(?:MODULE\.bazel|source\.json)"\s*:'
)

# Emails with an allowlist of benign / RFC-2606 reserved addresses.
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_EMAIL_ALLOW_EXACT = frozenset({"noreply@bytedance.com"})
_EMAIL_ALLOW_DOMAINS = ("pytorch.org", "example.com", "example.org")


@dataclass
class Finding:
    """A single flagged token on a line."""

    path: str
    lineno: int
    category: str
    preview: str


def _is_binary(data: bytes) -> bool:
    """Treat a file as binary if it contains a NUL byte in its head."""
    return b"\x00" in data[:8192]


def _octets_ok(match: str) -> bool:
    """Every octet must be in [0, 255] for a plausible IPv4 address."""
    return all(0 <= int(octet) <= 255 for octet in match.split("."))


def _is_routable_ip(match: str) -> bool:
    """Return True only for a routable, non-documentation IPv4 address.

    Excludes loopback, 0.0.0.0, RFC-1918 private, link-local, and the
    RFC-5737 documentation ranges so that only real public addresses trip
    the guard.
    """
    if not _octets_ok(match):
        return False
    a, b, c, _d = (int(x) for x in match.split("."))
    if a == 0 or a == 127:  # unspecified / loopback
        return False
    if a == 10:  # RFC-1918
        return False
    if a == 172 and 16 <= b <= 31:  # RFC-1918
        return False
    if a == 192 and b == 168:  # RFC-1918
        return False
    if a == 169 and b == 254:  # link-local
        return False
    if a == 192 and b == 0 and c == 2:  # RFC-5737 doc (TEST-NET-1)
        return False
    if a == 198 and b == 51 and c == 100:  # RFC-5737 doc (TEST-NET-2)
        return False
    if a == 203 and b == 0 and c == 113:  # RFC-5737 doc (TEST-NET-3)
        return False
    return True


def _looks_like_version(line: str, start: int, end: int) -> bool:
    """Reject dotted version strings that a bare IPv4 regex would match.

    A version-context guard: if the dotted-number run is immediately adjacent
    to ``==``, ``>=``, ``-``, or an alphabetic character (e.g.
    ``nvidia-curand==10.4.0.35`` or a driver token), it is a version, not an
    address. Official BCR metadata keys also place the version between a valid
    module name and a fixed metadata filename.
    """
    before = line[:start]
    after = line[end:]
    if before.endswith(("==", ">=", "<=", "~=", "-")) or before[-1:].isalpha():
        return True
    if after[:1].isalpha() or after.startswith(("-", "==", ">=")):
        return True
    bcr_key = _BCR_MODULE_METADATA_KEY_RE.match(line)
    if bcr_key is not None and bcr_key.span("version") == (start, end):
        return True
    return False


def _scan_line(path: str, lineno: int, line: str) -> Iterator[Finding]:
    """Yield findings for a single line, honoring the inline pragma."""
    if ALLOW_PRAGMA in line:
        return

    for match in _ABS_HOME_RE.finditer(line):
        token = match.group(0)
        if token == _HOME_PLACEHOLDER:
            continue
        yield Finding(path, lineno, "abs-home-path", token)

    for match in _IPV4_CANDIDATE_RE.finditer(line):
        token = match.group(0)
        if not _octets_ok(token):
            continue
        if _looks_like_version(line, match.start(), match.end()):
            continue
        if _is_routable_ip(token):
            yield Finding(path, lineno, "routable-ipv4", token)

    for match in _MAC_RE.finditer(line):
        yield Finding(path, lineno, "mac-address", match.group(0))

    for match in _HOSTNAME_RE.finditer(line):
        yield Finding(path, lineno, "hostname", match.group(0))

    for match in _GPU_UUID_RE.finditer(line):
        yield Finding(path, lineno, "gpu-uuid", match.group(0))

    for match in _KERNEL_TAG_RE.finditer(line):
        yield Finding(path, lineno, "kernel-tag", match.group(0))

    for match in _IDENTITY_RE.finditer(line):
        yield Finding(path, lineno, "identity", match.group(0))

    for match in _EMAIL_RE.finditer(line):
        token = match.group(0)
        if token in _EMAIL_ALLOW_EXACT:
            continue
        domain = token.rsplit("@", 1)[-1]
        if domain.endswith(_EMAIL_ALLOW_DOMAINS):
            continue
        yield Finding(path, lineno, "email", token)


def scan_file(path: str) -> list[Finding]:
    """Scan a single file, returning findings; allowlisted files return none."""
    if path in ALLOWLISTED_PATHS:
        return []
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except (FileNotFoundError, IsADirectoryError):
        return []
    if _is_binary(data):
        return []
    text = data.decode("utf-8", errors="replace")

    findings: list[Finding] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        findings.extend(_scan_line(path, lineno, line))
    return findings


def _git_tracked_files() -> list[str]:
    """Return every git-tracked path for the ``--all`` sweep."""
    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _resolve_paths(argv: list[str]) -> list[str]:
    """Resolve the argv into the list of files to scan."""
    if argv == ["--all"]:
        return _git_tracked_files()
    if "--all" in argv:
        raise ValueError("--all may not be combined with explicit file paths")
    return argv


def main(argv: Iterable[str]) -> int:
    paths = _resolve_paths(list(argv))

    all_findings: list[Finding] = []
    for path in paths:
        all_findings.extend(scan_file(path))

    for finding in all_findings:
        print(
            f"{finding.path}:{finding.lineno}: {finding.category}: "
            f"{finding.preview}"
        )

    if all_findings:
        print(
            f"\ncheck-no-pii: found {len(all_findings)} disallowed token(s). "
            f"Redact them, or add a '{ALLOW_PRAGMA}' pragma on the line if the "
            "token is an intentional fixture.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
