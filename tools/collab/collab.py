#!/usr/bin/env python3
"""Local cooperative Git-ticket coordination. Python standard library + Git.

Not an agent runtime or sandbox. Commands mutate local state only; never push.
See README.md for the trust, crash-recovery and ticket-format limitations.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import uuid

# Shared repository helper is a runtime dependency when copying this CLI.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from temp_root import temporary_root


class Refused(Exception):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def valid_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", value):
        raise Refused(f"Invalid identifier: {value!r}")
    return value


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    # Git and check children must resolve the repository from their assigned cwd.
    for key in list(environment):
        if key.startswith("GIT_") and key not in ("GIT_SSH_COMMAND", "GIT_SSH"):
            environment.pop(key)
    environment["GIT_TERMINAL_PROMPT"] = "0"
    return environment


def output_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    forbidden = Path("/") / "tmp"
    if path == forbidden or forbidden in path.parents or path == Path("/"):
        raise Refused("Output paths must be outside the system temporary directory and filesystem root")
    return path


def failpoint(name: str) -> None:
    # Test-only process death; deliberately bypass Python cleanup.
    if os.environ.get("COLLAB_FAILPOINT") == name:
        os._exit(86)


def ticket_sections(raw: bytes) -> tuple[bytes, bytes, bytes]:
    delimiters = re.finditer(rb"(?m)^---[ \t]*\r?$", raw)
    opening, closing = next(delimiters, None), next(delimiters, None)
    if opening is None or closing is None or raw[:opening.start()].strip():
        raise Refused("Ticket needs standalone --- front matter delimiters")
    return raw[:opening.end()], raw[opening.end():closing.start()], raw[closing.start():]


def parse_ticket(raw: bytes, expected_id: str) -> dict:
    """Read the deliberately narrow tk-compatible header subset.

    Required: id, status, deps (inline list). Unknown scalar fields are allowed.
    This is NOT a general YAML parser; multiline/block metadata is refused.
    """
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Refused("Ticket must be UTF-8") from exc
    _, header, body = ticket_sections(raw)
    fields = {}
    for line in header.decode("utf-8").strip().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        if line[:1].isspace() or ":" not in line:
            raise Refused("Only scalar/inline-list front matter is supported")
        key, value = line.split(":", 1)
        if key in fields or not re.fullmatch(r"[a-z][a-z_-]*", key):
            raise Refused("Duplicate or unsupported metadata key")
        value = value.strip()
        if value in ("|", ">", "|-", ">-"):
            raise Refused("Multiline metadata is unsupported")
        fields[key] = value
    ticket_id = fields.get("id", "").strip("\"'")
    status = fields.get("status", "").strip("\"'")
    if ticket_id != expected_id:
        raise Refused("Ticket filename/id mismatch")
    if status not in ("open", "in_progress", "closed"):
        raise Refused("Ticket status must be open, in_progress, or closed")
    deps = fields.get("deps", "")
    if not deps.startswith("[") or not deps.endswith("]"):
        raise Refused("deps must be an inline list, e.g. [T-1, T-2]")
    deps = [valid_id(x.strip().strip("\"'")) for x in deps[1:-1].split(",") if x.strip()]
    if len(deps) != len(set(deps)) or expected_id in deps:
        raise Refused("Duplicate or self dependency")
    title = next((line[2:].strip() for line in body.decode("utf-8").splitlines() if line.startswith("# ")), expected_id)
    return {"id": ticket_id, "status": status, "deps": deps, "title": title,
            "sha256": digest(raw)}


SCHEMA = """
CREATE TABLE config (id INTEGER PRIMARY KEY CHECK(id=1), value TEXT NOT NULL);
CREATE TABLE attempts (
 id TEXT PRIMARY KEY, ticket TEXT NOT NULL, owner TEXT NOT NULL,
 state TEXT NOT NULL, base TEXT NOT NULL, ticket_hash TEXT NOT NULL,
 path TEXT NOT NULL UNIQUE, branch TEXT NOT NULL UNIQUE,
 candidate TEXT, integration TEXT, check_path TEXT, log TEXT,
 created TEXT NOT NULL, updated TEXT NOT NULL
);
CREATE UNIQUE INDEX one_active_ticket ON attempts(ticket)
 WHERE state NOT IN ('landed','abandoned');
CREATE TABLE events (
 seq INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL,
 attempt TEXT, kind TEXT NOT NULL, details TEXT NOT NULL
);
"""

UNRESOLVED_LANDING = ("landing_prepared", "landing_started", "landing")


class Collab:
    def __init__(self, repo: str | Path):
        self.invoked = output_path(repo)
        common = self.git("rev-parse", "--path-format=absolute", "--git-common-dir", cwd=self.invoked).strip()
        self.common = output_path(common)
        self.home = self.common / "collab"
        self.db_path = self.home / "state.sqlite3"
        self.fds: list[int] = []
        self.config = None
        if self.home.exists():
            if not self.db_path.is_file():
                raise Refused("Partial initialization; inspect, do not overwrite")
            with self.db() as con:
                row = con.execute("SELECT value FROM config WHERE id=1").fetchone()
                if row:
                    self.config = json.loads(row[0])

    def git(self, *args: str, cwd: Path | None = None, input: str | None = None,
            check: bool = True, env: dict | None = None, raw: bool = False) -> str | bytes:
        if cwd is None:
            self.primary_identity()
        root = cwd or Path(self.config["root"])
        e = child_environment()
        if env:
            e.update(env)
        cmd = ["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false",
               "-c", "core.logAllRefUpdates=true", "-C", str(root), *args]
        p = subprocess.run(cmd, input=input, text=not raw, capture_output=True,
                           env=e, pass_fds=tuple(getattr(self, "fds", [])))
        if check and p.returncode:
            raise Refused(f"Git refused {' '.join(args[:3])}: {p.stderr.strip()}")
        return p.stdout

    @contextmanager
    def db(self):
        con = sqlite3.connect(self.db_path, timeout=5, isolation_level=None)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=5000")
        con.execute("PRAGMA synchronous=FULL")
        try:
            yield con
        finally:
            con.close()

    @contextmanager
    def tx(self):
        with self.db() as con:
            con.execute("BEGIN IMMEDIATE")
            try:
                yield con
                con.commit()
            except BaseException:
                con.rollback()
                raise

    @contextmanager
    def lock(self, name: str):
        if not self.home.exists():
            raise Refused("Run init first")
        with (self.home / (name + ".lock")).open("a+") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise Refused(f"{name} operation busy; retry later") from exc
            self.fds.append(f.fileno())
            try:
                yield
            finally:
                self.fds.remove(f.fileno())
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def event(self, con, attempt: str | None, kind: str, **details):
        con.execute("INSERT INTO events(at,attempt,kind,details) VALUES(?,?,?,?)",
                    (now(), attempt, kind, json.dumps(details, sort_keys=True)))

    def require(self):
        if self.config is None:
            raise Refused("Run init first; never delete state.sqlite3 to reset active work")

    def row(self, aid: str) -> dict:
        self.require()
        valid_id(aid)
        with self.db() as con:
            r = con.execute("SELECT * FROM attempts WHERE id=?", (aid,)).fetchone()
            receipt = con.execute("SELECT details FROM events WHERE attempt=? AND kind='check_receipt' ORDER BY seq DESC LIMIT 1",
                                  (aid,)).fetchone()
        if r is None:
            raise Refused("Unknown attempt")
        return {**dict(r), "check_receipt": json.loads(receipt[0]) if receipt else None}

    def change(self, aid: str, state: str, **fields):
        permitted = {"base", "candidate", "integration", "check_path", "log"}
        if not set(fields) <= permitted:
            raise Refused("Invalid state update")
        with self.tx() as con:
            values = {"state": state, "updated": now(), **fields}
            con.execute("UPDATE attempts SET " + ",".join(k+"=?" for k in values) + " WHERE id=?",
                        (*values.values(), aid))
            self.event(con, aid, state, **fields)

    def trunk(self) -> str:
        self.require()
        return self.git("rev-parse", "--verify", self.config["trunk_ref"] + "^{commit}").strip()

    def clean(self, path: Path, include_ignored: bool = False):
        if self.hidden_index(path):
            raise Refused(f"Workspace has hidden index entries: {path}; preserve files and classify flags manually")
        args = ["status", "--porcelain=v1", "-z", "--untracked-files=all"]
        if include_ignored:
            args.append("--ignored")
        if self.git(*args, cwd=path):
            raise Refused(f"Workspace is not clean: {path}; preserve/classify files manually")

    def hidden_index(self, path: Path) -> bool:
        # Lowercase -v tags mean assume-unchanged; S/s means skip-worktree.
        entries = self.git("ls-files", "-v", "-z", cwd=path).split("\0")
        return any(entry and (entry[0].islower() or entry[0] == "S") for entry in entries)

    def primary(self):
        self.primary_identity()
        root = Path(self.config["root"])
        ref = self.git("symbolic-ref", "-q", "HEAD", cwd=root, check=False).strip()
        if ref != self.config["trunk_ref"]:
            raise Refused("Primary checkout must be on configured trunk; no automatic switching")
        self.clean(root)

    def primary_identity(self):
        root = Path(self.config["root"])
        try:
            common = Path(self.git("rev-parse", "--path-format=absolute", "--git-common-dir", cwd=root).strip())
            top = Path(self.git("rev-parse", "--show-toplevel", cwd=root).strip())
            if root.resolve() != root or common.resolve() != self.common or top != root:
                raise Refused("repository or top-level mismatch")
        except (Refused, OSError) as exc:
            raise Refused(f"Primary checkout identity mismatch: {root}: {exc}") from exc

    def workspace(self, a, name="path", expected_head=None):
        """Validate identity before touching any registered attempt workspace."""
        path = Path(a[name])
        try:
            if path.resolve() != path or not path.is_dir():
                raise Refused("path moved or is a symlink")
            common = Path(self.git("rev-parse", "--path-format=absolute", "--git-common-dir", cwd=path).strip())
            top = Path(self.git("rev-parse", "--show-toplevel", cwd=path).strip())
            head = self.git("rev-parse", "HEAD", cwd=path).strip()
            branch = self.git("symbolic-ref", "-q", "HEAD", cwd=path, check=False).strip()
            expected_branch = "refs/heads/" + a["branch"] if name == "path" else ""
            records = self.git("worktree", "list", "--porcelain", "-z").split("\0\0")
            entries = [record.split("\0") for record in records if record]
            registered = next((entry for entry in entries if entry[0] == "worktree " + str(path)), [])
            if (common.resolve() != self.common or top != path or branch != expected_branch or
                    "HEAD " + head not in registered or
                    (expected_branch and "branch " + expected_branch not in registered) or
                    (not expected_branch and "detached" not in registered) or
                    (expected_head is not None and head != expected_head)):
                raise Refused("repository, registration, branch or HEAD mismatch")
        except (Refused, OSError) as exc:
            raise Refused(f"Workspace identity mismatch: {path}: {exc}") from exc
        return path

    def ticket(self, tid: str, commit: str | None = None) -> tuple[dict, bytes]:
        valid_id(tid)
        revision = commit or self.trunk()
        path = ".tickets/" + tid + ".md"
        entry = self.git("ls-tree", revision, "--", path).strip()
        if not entry or not entry.startswith("100644 blob "):
            raise Refused(f"Missing/non-regular committed ticket: {tid}")
        raw = self.git("show", revision + ":" + path, raw=True)
        return parse_ticket(raw, tid), raw

    def deps_ok(self, tid: str, commit: str):
        visiting, done = set(), set()
        def visit(t):
            if t in visiting:
                raise Refused("Dependency cycle")
            if t in done:
                return
            visiting.add(t)
            meta, _ = self.ticket(t, commit)
            for parent in meta["deps"]:
                visit(parent)
            visiting.remove(t)
            done.add(t)
        visit(tid)
        meta, _ = self.ticket(tid, commit)
        for parent in meta["deps"]:
            p, _ = self.ticket(parent, commit)
            if p["status"] != "closed":
                raise Refused(f"Dependency not closed on trunk: {parent}")

    def same_contract(self, a, commit: str):
        meta, _ = self.ticket(a["ticket"], commit)
        if meta["sha256"] != a["ticket_hash"]:
            raise Refused("Ticket changed since claim; explicitly abandon/re-admit instead")
        self.deps_ok(a["ticket"], commit)

    def no_uncertain_landing(self):
        with self.db() as con:
            r = con.execute("SELECT id FROM attempts WHERE state IN (?,?,?) LIMIT 1",
                            UNRESOLVED_LANDING).fetchone()
        if r:
            raise Refused(f"Unresolved landing {r['id']}; reconcile first")

    def init(self, trunk: str, check_argv: list[str], workspace_root: str | None = None):
        if self.db_path.exists():
            raise Refused("Already initialized or partial initialization; inspect, do not overwrite")
        if (not isinstance(check_argv, list) or not check_argv or
                not all(isinstance(x, str) and x and "\0" not in x for x in check_argv)):
            raise Refused("Provide a nonempty check argv JSON array")
        root = Path(self.git("rev-parse", "--show-toplevel", cwd=self.invoked).strip())
        trunk_ref = "refs/heads/" + trunk
        self.git("check-ref-format", trunk_ref, cwd=root)
        if self.git("symbolic-ref", "-q", "HEAD", cwd=root).strip() != trunk_ref:
            raise Refused("Initialize from the intended trunk checkout")
        self.git("rev-parse", "--verify", trunk_ref+"^{commit}", cwd=root)
        self.clean(root)
        work = output_path(workspace_root) if workspace_root else root.parent / (root.name + "-collab-worktrees")
        if work == root or root in work.parents:
            raise Refused("Workspaces must be outside the primary checkout")
        self.home.mkdir(mode=0o700, parents=False, exist_ok=False)
        self.config = {"schema": 1, "root": str(root), "trunk_ref": trunk_ref,
                       "workspace_root": str(work), "check_argv": check_argv}
        with self.lock("meta"):
            with self.db() as con:
                # Conservative rollback journal for the tiny prototype. WAL is optional,
                # not required; all writes are short and concurrency is low.
                con.execute("PRAGMA journal_mode=DELETE")
                con.executescript(SCHEMA)
                con.execute("INSERT INTO config VALUES(1,?)", (json.dumps(self.config),))
                self.event(con, None, "initialized", **self.config)
        return self.config

    def new(self, tid: str, title: str, deps: list[str]):
        self.require()
        valid_id(tid)
        deps = [valid_id(x) for x in deps]
        if len(deps) != len(set(deps)) or tid in deps:
            raise Refused("Duplicate or self dependency")
        with self.lock("meta"):
            self.primary()
            directory = Path(self.config["root"]) / ".tickets"
            if directory.is_symlink():
                raise Refused("Ticket directory cannot be a symlink")
            directory.mkdir(exist_ok=True)
            path = directory / (tid + ".md")
            text = (f"---\nid: {tid}\nstatus: open\ndeps: [{', '.join(deps)}]\n---\n\n# {title}\n\n"
                    "## Outcome\nDescribe the outcome and why it matters.\n\n"
                    "## Invariants\nList actual constraints, not implementation guesses.\n\n"
                    "## Allowed Outputs\nList exact files or directories a worker may write.\n\n"
                    "## Acceptance\nDescribe observable evidence.\n\n"
                    "## Discretion\nYou own the engineering strategy. Escalate changes to acceptance or authority.\n")
            with path.open("x") as f:
                f.write(text)
        return {"path": str(path), "next": "Edit and explicitly git-add/commit this ticket on trunk before claiming"}

    def claim(self, tid: str, owner: str):
        self.require()
        if not owner.strip():
            raise Refused("Owner required")
        with self.lock("meta"):
            base = self.trunk()
            meta, _ = self.ticket(tid, base)
            if meta["status"] == "closed":
                raise Refused("Ticket is closed")
            self.deps_ok(tid, base)
            aid = "a-" + uuid.uuid4().hex[:16]
            branch = f"collab/{tid}/{aid}"
            path = Path(self.config["workspace_root"]) / aid
            with self.tx() as con:
                try:
                    con.execute("INSERT INTO attempts(id,ticket,owner,state,base,ticket_hash,path,branch,created,updated) VALUES(?,?,?,?,?,?,?,?,?,?)",
                                (aid, tid, owner, "allocating", base, meta["sha256"], str(path), branch, now(), now()))
                except sqlite3.IntegrityError as exc:
                    raise Refused("Ticket already has an active claim") from exc
                self.event(con, aid, "allocating", ticket=tid, owner=owner, base=base)
            failpoint("after_claim_record")
            path.parent.mkdir(parents=True, exist_ok=True)
            self.git("worktree", "add", "--lock", "--reason", f"collab:{aid}", "-b", branch, str(path), base)
            failpoint("after_worktree_add")
            self.change(aid, "working")
        return self.row(aid)

    def submit(self, aid: str):
        with self.lock("meta"):
            a = self.row(aid)
            if a["state"] not in ("working", "submitted", "check_failed", "verified"):
                raise Refused("Attempt cannot submit in its current state")
            path = self.workspace(a)
            self.clean(path)
            if self.git("symbolic-ref", "-q", "HEAD", cwd=path).strip() != "refs/heads/"+a["branch"]:
                raise Refused("Worker changed its assigned branch")
            base, candidate = self.trunk(), self.git("rev-parse", "HEAD", cwd=path).strip()
            if base != a["base"]:
                raise Refused("Trunk moved; stop/abandon, claim fresh, and cherry-pick only the prior contribution")
            self.same_contract(a, base)
            if self.git("merge-base", base, candidate).strip() != base:
                raise Refused("Candidate diverges from its admitted base; stop/abandon, claim fresh, cherry-pick contribution")
            if base == candidate:
                raise Refused("Empty contribution")
            if self.git("rev-list", "--merges", f"{base}..{candidate}").strip():
                raise Refused("Use a linear private branch for this prototype")
            changed = self.git("diff", "--name-only", "-z", base, candidate).split("\0")
            if not any(changed):
                raise Refused("Empty contribution: candidate has no file changes")
            protected = (
                ".tickets/",
                ".github/",
                ".claude/",
                "docs/agents/",
                "tools/collab/",
            )
            root_protected = {"AGENTS.md", "CLAUDE.md", "CONSTITUTION.md"}
            if any(p in root_protected or p.startswith(protected) for p in changed if p):
                raise Refused("Candidate changes coordination/policy paths; use a separately approved operator change")
            self.git("update-ref", f"refs/collab/candidates/{aid}/{candidate}", candidate)
            self.change(aid, "submitted", base=base, candidate=candidate, integration=None)
        return self.row(aid)

    def check(self, aid: str):
        # The lane lock serializes final checks, not all implementation or DB writes.
        with self.lock("lane"):
            with self.lock("meta"):
                self.no_uncertain_landing()
                a = self.row(aid)
                if a["state"] not in ("submitted", "check_failed", "verified"):
                    raise Refused("Submit a candidate first")
                if self.trunk() != a["base"]:
                    raise Refused("Trunk moved; stop/abandon, claim fresh, cherry-pick contribution, submit/check again")
                self.same_contract(a, a["base"])
                tag = uuid.uuid4().hex[:12]
                path = Path(self.config["workspace_root"]) / f"check-{aid}-{tag}"
                log = self.home / f"check-{aid}-{tag}.log"
                self.change(aid, "checking", check_path=str(path), log=str(log), integration=None)
                self.git("worktree", "add", "--detach", "--lock", "--reason", f"collab-check:{aid}", str(path), a["candidate"])
                # Only the fresh detached verification worktree is rewritten.
                self.git("reset", "--soft", a["base"], cwd=path)
                tpath = path / ".tickets" / (a["ticket"] + ".md")
                raw = tpath.read_bytes()
                if digest(raw) != a["ticket_hash"]:
                    raise Refused("Candidate changed ticket bytes")
                prefix, header, suffix = ticket_sections(raw)
                header, count = re.subn(rb"(?m)^status:[^\r\n]*", b"status: closed", header)
                if count != 1:
                    raise Refused("Expected one status field")
                tpath.write_bytes(prefix + header + suffix)
                self.git("add", "--", ".tickets/"+a["ticket"]+".md", cwd=path)
                message = f"{a['ticket']}: accepted contribution\n\nCollab-Attempt: {aid}\nWorker-Candidate: {a['candidate']}\n"
                self.git("commit", "-m", message, cwd=path)
                integration = self.git("rev-parse", "HEAD", cwd=path).strip()
                self.git("update-ref", f"refs/collab/checks/{aid}/{tag}", integration)
                self.change(aid, "checking", integration=integration)
            failpoint("before_check_command")
            argv = self.config["check_argv"]
            code = -1
            with log.open("x") as stream:
                stream.write(json.dumps({"attempt": aid, "subject": integration, "argv": argv, "started": now()}) + "\n")
                stream.flush()
                try:
                    p = subprocess.run(argv, cwd=path, stdout=stream, stderr=subprocess.STDOUT,
                                       env=child_environment(),
                                       pass_fds=tuple(self.fds))
                    code = p.returncode
                except OSError as exc:
                    stream.write(str(exc)+"\n")
                stream.flush()
                os.fsync(stream.fileno())
            with self.lock("meta"):
                current = self.row(aid)
                if current["state"] != "checking":
                    raise Refused("Check state changed concurrently")
                clean = same = identity = False
                errors = []
                try:
                    self.workspace({**a, "check_path": str(path)}, "check_path", integration)
                    identity = True
                except (Refused, OSError, ValueError) as exc:
                    errors.append(str(exc))
                try:
                    clean = (self.git("status", "--porcelain=v1", "-z", "--untracked-files=all", cwd=path) == ""
                             and not self.hidden_index(path))
                    same = self.git("rev-parse", "HEAD", cwd=path).strip() == integration
                except (Refused, OSError, ValueError) as exc:
                    clean = same = identity = False
                    errors.append(str(exc))
                target = None
                try:
                    target = self.trunk()
                except (Refused, OSError, ValueError) as exc:
                    errors.append(str(exc))
                passed = code == 0 and clean and same and identity and target == a["base"]
                log_hash = digest(log.read_bytes())
                with self.tx() as con:
                    state = "verified" if passed else "check_failed"
                    con.execute("UPDATE attempts SET state=?,updated=? WHERE id=?", (state, now(), aid))
                    self.event(con, aid, state)
                    self.event(con, aid, "check_receipt", subject=integration, argv=argv,
                               exit_code=code, clean=clean, same_head=same, identity=identity,
                               base=a["base"], candidate=a["candidate"], check_path=str(path),
                               log=str(log), trunk=target,
                               inspection_error="; ".join(errors) if errors else None,
                               log_sha256=log_hash, passed=passed)
                    failpoint("before_check_receipt_commit")
                failpoint("after_check_receipt_commit")
            if not passed:
                raise Refused(f"Check failed, changed its workspace, or trunk moved; log: {log}")
        return self.row(aid)

    def finish_landing(self, a):
        ref = f"refs/collab/landed/{a['id']}"
        old = self.git("rev-parse", "--verify", ref, check=False).strip()
        if old and old != a["integration"]:
            raise Refused("Conflicting landing receipt ref")
        if not old:
            self.git("update-ref", ref, a["integration"], "")
        self.change(a["id"], "landed")

    def require_receipt(self, a):
        receipt = a["check_receipt"]
        expected = {"subject": a["integration"], "base": a["base"], "candidate": a["candidate"],
                    "check_path": a["check_path"], "log": a["log"], "trunk": a["base"],
                    "argv": self.config["check_argv"], "exit_code": 0, "clean": True,
                    "same_head": True, "identity": True, "passed": True}
        if not receipt or any(receipt.get(k) != v for k, v in expected.items()):
            raise Refused("Missing/mismatched durable check receipt; submit/check again")
        if digest(Path(a["log"]).read_bytes()) != receipt.get("log_sha256"):
            raise Refused("Check log changed; durable receipt no longer matches")

    def land(self, aid: str):
        # This explicit human operator action is the manual review/approval gate.
        with self.lock("lane"), self.lock("meta"):
            self.no_uncertain_landing()
            a = self.row(aid)
            if a["state"] != "verified":
                raise Refused("Exact integration candidate has not passed checking")
            self.require_receipt(a)
            self.primary()
            if self.trunk() != a["base"]:
                raise Refused("Trunk moved; stop/abandon, claim fresh, cherry-pick contribution, submit/check again")
            self.same_contract(a, a["base"])
            if self.git("rev-parse", a["integration"]+"^").strip() != a["base"]:
                raise Refused("Integration candidate must have the expected parent")
            self.change(aid, "landing_prepared")
            failpoint("before_git_land")
            self.change(aid, "landing_started")
            failpoint("after_land_started")
            # Unlike update-ref on a checked-out branch, this safely updates index/files.
            self.git("merge", "--ff-only", "--no-edit", "--no-overwrite-ignore", a["integration"],
                     env={"GIT_REFLOG_ACTION": f"collab-land:{aid}"})
            failpoint("after_git_land")
            if self.trunk() != a["integration"]:
                raise Refused("Unexpected target after merge; reconcile")
            self.finish_landing(a)
        return self.row(aid)

    def recover(self, aid: str, confirmed: bool):
        if not confirmed:
            raise Refused("Requires --confirm-stopped: inspect old CLI/Git/check processes first")
        with self.lock("lane"), self.lock("meta"):
            a = self.row(aid)
            if a["state"] in UNRESOLVED_LANDING:
                self.primary()
                self.require_receipt(a)
                log = self.git("reflog", "show", "--format=%H%x00%gs", self.config["trunk_ref"])
                entries = log.splitlines()
                attributed = any(line.startswith(a["integration"]+"\0collab-land:"+aid+":") and
                                 i + 1 < len(entries) and entries[i+1].split("\0", 1)[0] == a["base"]
                                 for i, line in enumerate(entries))
                target = self.trunk()
                if (a["state"] != "landing_prepared" and attributed and
                        self.git("merge-base", a["integration"], target).strip() == a["integration"]):
                    self.finish_landing(a)
                elif a["state"] == "landing_prepared" and target == a["base"] and not attributed:
                    self.change(aid, "verified")
                else:
                    raise Refused("Ambiguous landing; preserve state and inspect manually")
            elif a["state"] == "allocating":
                path = Path(a["path"])
                if not path.exists():
                    # No deletion or new execution. Claim stays reserved pending operator choice.
                    raise Refused("Workspace not present; explicitly abandon before a new claim")
                self.workspace(a, expected_head=a["base"])
                self.change(aid, "working")
            elif a["state"] == "checking":
                self.change(aid, "check_failed")
            else:
                return {"attempt": aid, "state": a["state"], "action": "none; resume existing work, do not relaunch automatically"}
        return self.row(aid)

    def abandon(self, aid: str, confirmed: bool):
        if not confirmed:
            raise Refused("Requires --confirm-stopped; this command does not stop processes")
        with self.lock("lane"), self.lock("meta"):
            a = self.row(aid)
            if a["state"] in (*UNRESOLVED_LANDING, "landed", "abandoned"):
                raise Refused("Cannot abandon this state; reconcile landing first")
            self.change(aid, "abandoned")
        return self.row(aid)

    def gc(self, aid: str, confirmed: bool):
        if not confirmed:
            raise Refused("Requires --confirm-stopped; no process supervision is provided")
        with self.lock("lane"), self.lock("meta"):
            a = self.row(aid)
            if a["state"] not in ("landed", "abandoned"):
                raise Refused("Only terminal attempts may be cleaned")
            removed = []
            paths = []
            for name in ("path", "check_path"):
                if not a[name]:
                    continue
                path = Path(a[name])
                if not path.exists():
                    continue
                if name == "check_path" and not a["integration"]:
                    raise Refused("Workspace identity unknown for incomplete check; preserve it manually")
                self.workspace(a, name, a["integration"] if name == "check_path" else a["candidate"])
                self.clean(path, include_ignored=True)
                paths.append(path)
            # Validate every workspace before unlocking or removing any of them.
            for path in paths:
                self.git("worktree", "unlock", str(path))
                try:
                    self.git("worktree", "remove", str(path))
                except Refused:
                    self.git("worktree", "lock", "--reason", f"collab:{aid}", str(path), check=False)
                    raise
                removed.append(str(path))
            # Keep branches and sealed refs: no hidden destructive pruning.
            with self.tx() as con:
                self.event(con, aid, "cleanup", removed=removed)
        return {"removed": removed, "retained": "branches, candidate/check refs, logs, database history"}

    def status(self):
        self.require()
        base = self.trunk()
        paths = self.git("ls-tree", "-r", "--name-only", base, "--", ".tickets/").splitlines()
        with self.db() as con:
            attempts = [dict(r) for r in con.execute("SELECT * FROM attempts ORDER BY created")]
        active = {a["ticket"]: a for a in attempts if a["state"] not in ("landed", "abandoned")}
        tickets = []
        for p in paths:
            if p.count("/") != 1 or not p.endswith(".md"):
                continue
            tid = Path(p).stem
            try:
                meta, _ = self.ticket(tid, base)
                blocker = None
                try:
                    self.deps_ok(tid, base)
                except Refused as exc:
                    blocker = str(exc)
                a = active.get(tid)
                tickets.append({**meta, "active_attempt": a["id"] if a else None,
                                "execution": a["state"] if a else None,
                                "ready": meta["status"] != "closed" and not blocker and not a,
                                "blocker": blocker})
            except Refused as exc:
                tickets.append({"id": tid, "error": str(exc), "ready": False})
        return {"trunk": base, "tickets": tickets, "attempts": attempts}

    def backup(self, destination: str):
        requested = Path(destination).expanduser()
        if requested.is_symlink():
            raise Refused("Backup destination exists as a symlink")
        target = output_path(destination)
        with self.lock("lane"), self.lock("meta"), self.db() as source:
            # O_EXCL arbitrates even backups from independent repositories.
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(fd)
            with sqlite3.connect(target) as dest:
                source.backup(dest)
        return {"database_backup": str(target), "note": "Also preserve Git refs, workspaces and logs; not a full-project checkpoint"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=".")
    sub = p.add_subparsers(dest="command", required=True)
    x = sub.add_parser("init"); x.add_argument("--trunk", required=True); x.add_argument("--check-json", required=True); x.add_argument("--workspace-root")
    x = sub.add_parser("new"); x.add_argument("ticket"); x.add_argument("title"); x.add_argument("--deps", nargs="*", default=[])
    x = sub.add_parser("claim"); x.add_argument("ticket"); x.add_argument("--owner", required=True)
    for name in ("show", "submit", "check", "land", "recover", "abandon", "gc"):
        x = sub.add_parser(name); x.add_argument("attempt")
        if name in ("recover", "abandon", "gc"):
            x.add_argument("--confirm-stopped", action="store_true")
    sub.add_parser("status")
    sub.add_parser("ready")
    x = sub.add_parser("backup"); x.add_argument("destination")
    a = p.parse_args(argv)
    try:
        temporary = temporary_root()
        os.environ.update(TMPDIR=str(temporary), TMP=str(temporary), TEMP=str(temporary))
        c = Collab(a.repo)
        if a.command == "init":
            out = c.init(a.trunk, json.loads(a.check_json), a.workspace_root)
        elif a.command == "new": out = c.new(a.ticket, a.title, a.deps)
        elif a.command == "claim": out = c.claim(a.ticket, a.owner)
        elif a.command == "show": out = c.row(a.attempt)
        elif a.command in ("recover", "abandon", "gc"):
            out = getattr(c, a.command)(a.attempt, a.confirm_stopped)
        elif a.command in ("submit", "check", "land"):
            out = getattr(c, a.command)(a.attempt)
        elif a.command == "ready": out = [t for t in c.status()["tickets"] if t["ready"]]
        elif a.command == "backup": out = c.backup(a.destination)
        else: out = c.status()
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0
    except (Refused, sqlite3.Error, OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
