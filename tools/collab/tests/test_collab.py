"""Public CLI proof using real disposable Git repositories and process death."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

CLI = Path(__file__).resolve().parents[1] / "collab.py"
sys.path.insert(0, str(CLI.parents[1]))
from temp_root import temporary_root

SYSTEM_TMP = str(Path("/") / "tmp")


class CollabTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="collab-test-", dir=temporary_root())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "project"
        self.repo.mkdir()
        self.env = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                        GIT_AUTHOR_NAME="Test", GIT_AUTHOR_EMAIL="test@example.invalid",
                        GIT_COMMITTER_NAME="Test", GIT_COMMITTER_EMAIL="test@example.invalid")
        self.git("init", "-b", "main")
        self.git("config", "user.name", "Collab Test")
        self.git("config", "user.email", "collab@example.invalid")
        (self.repo / "code.txt").write_text("base\n")
        (self.repo / ".gitignore").write_text("build/\n")
        self.commit("initial")

    def run_process(self, argv, cwd=None, env=None):
        return subprocess.run(argv, cwd=cwd or self.repo, env=env or self.env,
                              text=True, capture_output=True)

    def git(self, *args, cwd=None, ok=True):
        result = self.run_process(["git", *args], cwd=cwd)
        if ok:
            self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip()

    def commit(self, message, cwd=None):
        self.git("add", "--all", cwd=cwd)
        self.git("commit", "-m", message, cwd=cwd)
        return self.git("rev-parse", "HEAD", cwd=cwd)

    def cli(self, *args, repo=None, ok=True, failpoint=None, env=None):
        child_env = dict(env or self.env)
        if failpoint:
            child_env["COLLAB_FAILPOINT"] = failpoint
        result = self.run_process([sys.executable, str(CLI), "--repo", str(repo or self.repo),
                                   *args], env=child_env)
        if ok:
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def init(self, argv=None):
        return self.cli("init", "--trunk", "main", "--check-json",
                        json.dumps(argv or [sys.executable, "-c", "print('checked')"]),
                        "--workspace-root", str(self.root / "workspaces"))

    def ticket(self, tid="T-one", deps=(), status="open", crlf=False):
        path = self.repo / ".tickets" / (tid + ".md")
        path.parent.mkdir(exist_ok=True)
        raw = (f"---\nid: {tid}\nstatus: {status}\ndeps: [{', '.join(deps)}]\n---\n\n# {tid}\n\nDo the thing.\n").encode()
        path.write_bytes(raw.replace(b"\n", b"\r\n") if crlf else raw)
        return path

    def claim(self, tid="T-one"):
        return self.cli("claim", tid, "--owner", "test-worker")

    def contribute(self, attempt, content="contribution\n"):
        path = Path(attempt["path"])
        (path / "code.txt").write_text(content)
        return self.commit("worker contribution", cwd=path)

    def prepared(self, argv=None):
        self.init(argv)
        self.ticket()
        self.commit("ticket")
        a = self.claim()
        self.contribute(a)
        return self.cli("submit", a["id"])

    def test_checked_commit_and_ticket_closure_land_together(self):
        a = self.prepared()
        checked = self.cli("check", a["id"])
        self.assertEqual(checked["state"], "verified")
        self.assertEqual(self.git("rev-parse", "HEAD"), a["base"])
        landed = self.cli("land", a["id"])
        self.assertEqual(landed["state"], "landed")
        self.assertEqual(self.git("rev-parse", "HEAD"), checked["integration"])
        self.assertEqual(self.git("rev-parse", "HEAD^"), a["base"])
        self.assertIn("status: closed", (self.repo / ".tickets/T-one.md").read_text())
        self.assertEqual((self.repo / "code.txt").read_text(), "contribution\n")

    def test_init_rejects_non_array_check_argv(self):
        for value in ('"true"', '{"true": "value"}', '[]', '[""]', '[1]', '["bad\\u0000arg"]'):
            with self.subTest(value=value):
                result = self.cli("init", "--trunk", "main", "--check-json", value, ok=False)
                self.assertIn("argv", result.stderr)
                self.assertFalse((self.repo / ".git/collab/state.sqlite3").exists())

    def test_claim_hashes_exact_crlf_ticket_and_check_preserves_line_endings(self):
        self.init()
        ticket = self.ticket(crlf=True)
        original = ticket.read_bytes()
        self.commit("CRLF ticket")
        a = self.claim()
        self.assertEqual(a["ticket_hash"], hashlib.sha256(original).hexdigest())
        self.contribute(a)
        self.cli("submit", a["id"])
        checked = self.cli("check", a["id"])
        self.assertEqual((Path(checked["check_path"]) / ".tickets/T-one.md").read_bytes(),
                         original.replace(b"status: open", b"status: closed"))

    def test_check_death_before_receipt_commit_cannot_publish_verified(self):
        a = self.prepared()
        result = self.cli("check", a["id"], failpoint="before_check_receipt_commit", ok=False)
        self.assertEqual(result.returncode, 86)
        current = self.cli("show", a["id"])
        self.assertEqual(current["state"], "checking")
        self.assertIsNone(current["check_receipt"])
        self.cli("land", a["id"], ok=False)
        self.assertEqual(self.cli("recover", a["id"], "--confirm-stopped")["state"], "check_failed")

    def test_check_receipt_is_durable_with_verified_state(self):
        a = self.prepared()
        result = self.cli("check", a["id"], failpoint="after_check_receipt_commit", ok=False)
        self.assertEqual(result.returncode, 86)
        checked = self.cli("show", a["id"])
        self.assertEqual(checked["state"], "verified")
        receipt = checked["check_receipt"]
        self.assertEqual(receipt["subject"], checked["integration"])
        self.assertEqual(receipt["base"], checked["base"])
        self.assertEqual(receipt["log_sha256"], hashlib.sha256(Path(checked["log"]).read_bytes()).hexdigest())
        self.assertTrue(receipt["passed"])
        self.assertEqual(self.cli("land", a["id"])["state"], "landed")

    def test_prepared_landing_at_base_can_recover_verified(self):
        a = self.prepared()
        self.cli("check", a["id"])
        result = self.cli("land", a["id"], failpoint="before_git_land", ok=False)
        self.assertEqual(result.returncode, 86)
        self.assertEqual(self.cli("show", a["id"])["state"], "landing_prepared")
        self.cli("recover", a["id"], ok=False)
        self.assertEqual(self.cli("recover", a["id"], "--confirm-stopped")["state"], "verified")
        self.assertEqual(self.cli("land", a["id"])["state"], "landed")

    def test_started_landing_without_attribution_stays_ambiguous_at_base(self):
        a = self.prepared()
        self.cli("check", a["id"])
        result = self.cli("land", a["id"], failpoint="after_land_started", ok=False)
        self.assertEqual(result.returncode, 86)
        self.assertEqual(self.cli("show", a["id"])["state"], "landing_started")
        self.assertEqual(self.git("rev-parse", "HEAD"), a["base"])
        result = self.cli("recover", a["id"], "--confirm-stopped", ok=False)
        self.assertIn("Ambiguous", result.stderr)
        self.cli("abandon", a["id"], "--confirm-stopped", ok=False)
        self.cli("land", a["id"], ok=False)

    def test_every_unresolved_landing_phase_blocks_other_check_and_land(self):
        self.init()
        self.ticket()
        self.ticket("T-two")
        self.commit("two tickets")
        first, second = self.claim(), self.claim("T-two")
        for a in (first, second):
            self.contribute(a, a["id"])
            self.cli("submit", a["id"])
            self.cli("check", a["id"])
        self.cli("land", first["id"], failpoint="before_git_land", ok=False)
        for command in ("check", "land"):
            result = self.cli(command, second["id"], ok=False)
            self.assertIn("Unresolved landing", result.stderr)
        self.cli("recover", first["id"], "--confirm-stopped")
        self.cli("land", first["id"], failpoint="after_land_started", ok=False)
        for command in ("check", "land"):
            result = self.cli(command, second["id"], ok=False)
            self.assertIn("Unresolved landing", result.stderr)

    def test_direct_cli_rejects_unsafe_temp_and_output_roots(self):
        for value in (SYSTEM_TMP, "relative", "/"):
            with self.subTest(value=value):
                result = self.cli("init", "--trunk", "main", "--check-json", '["true"]',
                                  env=dict(self.env, TMPDIR=value), ok=False)
                self.assertIn("TMPDIR", result.stderr)
                self.assertFalse((self.repo / ".git/collab/state.sqlite3").exists())
        result = self.cli("init", "--trunk", "main", "--check-json", '["true"]',
                          "--workspace-root", SYSTEM_TMP + "/collab-must-not-create", ok=False)
        self.assertIn("temporary", result.stderr)
        self.init()
        self.cli("backup", SYSTEM_TMP + "/collab-must-not-create.sqlite3", ok=False)

    def test_claim_uses_collab_branch_and_child_inherits_safe_temp(self):
        code = ("import os,tempfile; from pathlib import Path; p=Path(tempfile.gettempdir()); "
                "assert p==Path(os.environ['TMPDIR']); "
                f"assert not p.is_relative_to(Path({SYSTEM_TMP!r})); print(p)")
        a = self.prepared([sys.executable, "-c", code])
        self.assertTrue(a["branch"].startswith("collab/"))
        self.assertEqual(self.cli("check", a["id"])["state"], "verified")

    def substitute_workspace(self, attempt):
        path = Path(attempt["path"])
        path.rename(path.with_name(path.name + "-preserved"))
        path.mkdir()
        self.git("init", "-b", attempt["branch"], cwd=path)
        (path / "unrelated.txt").write_text("valuable\n")
        self.commit("unrelated repository", cwd=path)
        return path

    def test_submit_rejects_other_repository_at_assigned_path(self):
        self.init()
        self.ticket()
        self.commit("ticket")
        a = self.claim()
        path = self.substitute_workspace(a)
        result = self.cli("submit", a["id"], ok=False)
        self.assertIn("identity", result.stderr)
        self.assertEqual((path / "unrelated.txt").read_text(), "valuable\n")

    def test_gc_rejects_substituted_repository_before_unlocking(self):
        a = self.prepared()
        self.cli("abandon", a["id"], "--confirm-stopped")
        path = self.substitute_workspace(a)
        result = self.cli("gc", a["id"], "--confirm-stopped", ok=False)
        self.assertIn("identity", result.stderr)
        self.assertTrue(path.exists())
        self.assertIn("locked collab:", self.git("worktree", "list", "--porcelain"))

    def test_gc_refuses_changed_check_head(self):
        a = self.prepared()
        checked = self.cli("check", a["id"])
        self.cli("abandon", a["id"], "--confirm-stopped")
        path = Path(checked["check_path"])
        self.git("commit", "--allow-empty", "-m", "check changed after verification", cwd=path)
        result = self.cli("gc", a["id"], "--confirm-stopped", ok=False)
        self.assertIn("identity", result.stderr)
        self.assertTrue(Path(a["path"]).exists())
        self.assertTrue(path.exists())

    def test_allocation_adoption_requires_registered_exact_base_head(self):
        self.init()
        self.ticket()
        self.commit("ticket")
        self.cli("claim", "T-one", "--owner", "worker", failpoint="after_worktree_add", ok=False)
        a = self.cli("status")["attempts"][0]
        self.git("commit", "--allow-empty", "-m", "unexpected moved HEAD", cwd=Path(a["path"]))
        result = self.cli("recover", a["id"], "--confirm-stopped", ok=False)
        self.assertIn("identity", result.stderr)
        self.assertEqual(self.cli("show", a["id"])["state"], "allocating")

    def test_racing_claims_have_one_durable_winner(self):
        self.init()
        self.ticket()
        self.commit("ticket")
        processes = [subprocess.Popen([sys.executable, str(CLI), "--repo", str(self.repo),
                                      "claim", "T-one", "--owner", owner], env=self.env,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                     for owner in ("worker-a", "worker-b")]
        results = [(process.wait(), *process.communicate()) for process in processes]
        self.assertEqual(sum(code == 0 for code, _, _ in results), 1, results)
        attempts = self.cli("status")["attempts"]
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["state"], "working")
        self.assertIn("locked collab:" + attempts[0]["id"], self.git("worktree", "list", "--porcelain"))

    def test_new_is_not_dispatchable_until_committed_and_worktrees_share_claims(self):
        self.init()
        authored = self.cli("new", "T-one", "An authored ticket")
        self.assertTrue(Path(authored["path"]).is_file())
        self.assertEqual(self.cli("ready"), [])
        self.cli("claim", "T-one", "--owner", "worker", ok=False)
        self.commit("authored ticket")
        self.assertEqual([ticket["id"] for ticket in self.cli("ready")], ["T-one"])
        a = self.claim()
        self.assertEqual(self.cli("show", a["id"], repo=Path(a["path"]))["id"], a["id"])
        self.assertEqual(self.cli("ready", repo=Path(a["path"])), [])
        self.cli("claim", "T-one", "--owner", "second", repo=Path(a["path"]), ok=False)
        self.cli("init", "--trunk", "main", "--check-json", '["true"]', ok=False)

    def test_dependencies_missing_cycles_closed_and_regular_ticket_modes(self):
        self.init()
        self.ticket("T-dep", status="closed")
        self.ticket("T-ready", deps=["T-dep"])
        self.ticket("T-blocked", deps=["T-ready"])
        self.ticket("T-missing", deps=["does-not-exist"])
        self.ticket("T-cycle-a", deps=["T-cycle-b"])
        self.ticket("T-cycle-b", deps=["T-cycle-a"])
        self.ticket("T-executable").chmod(0o755)
        self.commit("dependency graph")
        self.assertEqual([t["id"] for t in self.cli("ready")], ["T-ready"])
        for tid in ("T-dep", "T-blocked", "T-missing", "T-cycle-a", "T-executable", "T-rea", "../T-ready"):
            with self.subTest(ticket=tid):
                self.cli("claim", tid, "--owner", "worker", ok=False)
        a = self.claim("T-ready")
        self.assertEqual(a["state"], "working")

    def test_stale_claim_cannot_submit_even_after_private_branch_fast_forward(self):
        self.init()
        self.ticket()
        self.commit("ticket")
        a = self.claim()
        (self.repo / "operator.txt").write_text("new trunk\n")
        self.commit("trunk advances")
        self.git("merge", "--ff-only", "main", cwd=Path(a["path"]))
        self.contribute(a)
        result = self.cli("submit", a["id"], ok=False)
        self.assertIn("Trunk moved", result.stderr)

    def test_landing_rejects_replaced_primary_checkout(self):
        self.ticket()
        self.commit("ticket")
        self.git("switch", "-c", "source")
        operator = self.root / "operator"
        self.git("worktree", "add", str(operator), "main")
        self.cli("init", "--trunk", "main", "--check-json", '["true"]',
                 "--workspace-root", str(self.root / "workspaces"), repo=operator)
        a = self.cli("claim", "T-one", "--owner", "worker", repo=operator)
        self.contribute(a)
        self.cli("submit", a["id"], repo=Path(a["path"]))
        self.cli("check", a["id"], repo=Path(a["path"]))
        operator.rename(self.root / "preserved-operator")
        self.git("clone", "--no-hardlinks", "--branch", "main", str(self.repo), str(operator))
        result = self.cli("land", a["id"], repo=Path(a["path"]), ok=False)
        self.assertIn("identity", result.stderr)
        self.assertEqual(self.git("rev-parse", "HEAD", cwd=operator), a["base"])
        self.assertEqual(self.cli("show", a["id"], repo=Path(a["path"]))["state"], "verified")

    def test_protected_coordination_and_policy_paths_cannot_submit(self):
        self.init()
        self.ticket()
        self.commit("ticket")
        a = self.claim()
        worker = Path(a["path"])
        for name in (
            ".github/workflow.yml",
            ".claude/CLAUDE.md",
            "docs/agents/policy.md",
            "tools/collab/local_edit.py",
            "AGENTS.md",
            "CLAUDE.md",
            "CONSTITUTION.md",
            ".tickets/T-extra.md",
        ):
            with self.subTest(path=name):
                target = worker / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("worker policy edit\n")
                self.commit("protected path", cwd=worker)
                result = self.cli("submit", a["id"], ok=False)
                self.assertIn("coordination/policy", result.stderr)
                target.unlink()
                self.commit("withdraw protected edit", cwd=worker)

    def test_dirty_worker_and_operator_files_are_preserved(self):
        a = self.prepared()
        worker = Path(a["path"])
        (worker / "code.txt").write_text("uncommitted worker work\n")
        self.cli("submit", a["id"], ok=False)
        self.assertEqual((worker / "code.txt").read_text(), "uncommitted worker work\n")
        checked = self.cli("check", a["id"])
        (self.repo / "code.txt").write_text("uncommitted operator work\n")
        self.cli("land", a["id"], ok=False)
        self.assertEqual((self.repo / "code.txt").read_text(), "uncommitted operator work\n")
        self.assertEqual(self.cli("show", a["id"])["state"], "verified")
        self.assertEqual(self.git("rev-parse", "HEAD"), checked["base"])

    def test_uncommitted_contract_is_ignored_but_committed_byte_change_invalidates_claim(self):
        a = self.prepared()
        ticket = self.repo / ".tickets/T-one.md"
        original = ticket.read_bytes()
        ticket.write_bytes(original + b"Uncommitted proposal.\n")
        checked = self.cli("check", a["id"])
        self.assertEqual(checked["state"], "verified")
        ticket.write_bytes(original.replace(b"\n", b"\r\n"))
        self.commit("contract line-ending change")
        for command in ("submit", "check", "land"):
            self.cli(command, a["id"], ok=False)
        self.assertEqual(self.cli("show", a["id"])["ticket_hash"], a["ticket_hash"])

    def test_failing_mutating_and_head_moving_checks_never_verify(self):
        mode_file = self.root / "check-mode"
        code = ("import subprocess,sys; from pathlib import Path; "
                f"mode=Path({str(mode_file)!r}).read_text(); "
                "print('mode='+mode); "
                "Path('code.txt').write_text('mutation') if mode=='tracked' else None; "
                "Path('extra.txt').write_text('untracked') if mode=='untracked' else None; "
                "subprocess.run(['git','commit','--allow-empty','-m','mutated HEAD'],check=True) if mode=='head' else None; "
                "sys.exit(7 if mode=='fail' else 0)")
        a = self.prepared([sys.executable, "-c", code])
        for mode in ("fail", "tracked", "untracked", "head"):
            with self.subTest(mode=mode):
                mode_file.write_text(mode)
                self.cli("check", a["id"], ok=False)
                current = self.cli("show", a["id"])
                self.assertEqual(current["state"], "check_failed")
                self.assertFalse(current["check_receipt"]["passed"])
                self.assertIn("mode=" + mode, Path(current["log"]).read_text())
                self.cli("land", a["id"], ok=False)
        mode_file.write_text("pass")
        checked = self.cli("check", a["id"])
        self.assertEqual(checked["state"], "verified")
        self.assertEqual(len(list((self.root / "workspaces").glob("check-*"))), 5)

    def test_check_uses_literal_argv_and_land_rejects_changed_receipt_log(self):
        marker = self.root / "shell-must-not-execute"
        literal = f"; touch {marker}; $(false)"
        argv = [sys.executable, "-c", "import sys; print(sys.argv[1])", literal]
        a = self.prepared(argv)
        checked = self.cli("check", a["id"])
        self.assertFalse(marker.exists())
        self.assertEqual(checked["check_receipt"]["argv"], argv)
        self.assertIn(literal, Path(checked["log"]).read_text())
        Path(checked["log"]).write_text("replaced verification output\n")
        result = self.cli("land", a["id"], ok=False)
        self.assertIn("receipt", result.stderr)
        self.assertEqual(self.git("rev-parse", "HEAD"), a["base"])

    def test_zero_exit_with_trunk_movement_is_check_failed(self):
        argv = [sys.executable, "-c", "import subprocess; subprocess.run(['git','-C'," +
                repr(str(self.repo)) + ",'commit','--allow-empty','-m','operator moved trunk'],check=True)"]
        a = self.prepared(argv)
        self.cli("check", a["id"], ok=False)
        current = self.cli("show", a["id"])
        self.assertEqual(current["state"], "check_failed")
        self.assertEqual(current["check_receipt"]["exit_code"], 0)
        self.assertTrue(current["check_receipt"]["clean"])
        self.assertNotEqual(current["check_receipt"]["trunk"], current["base"])
        self.cli("land", a["id"], ok=False)

    def test_refresh_abandons_stale_attempt_and_cherry_picks_into_fresh_claim(self):
        a = self.prepared()
        self.cli("check", a["id"])
        (self.repo / "operator.txt").write_text("new trunk\n")
        new_base = self.commit("advance trunk")
        for command in ("submit", "check", "land"):
            self.cli(command, a["id"], ok=False)
        self.cli("abandon", a["id"], ok=False)
        self.cli("abandon", a["id"], "--confirm-stopped")
        self.cli("submit", a["id"], ok=False)
        fresh = self.claim()
        self.assertEqual(fresh["base"], new_base)
        self.assertNotEqual(fresh["path"], a["path"])
        self.git("cherry-pick", a["candidate"], cwd=Path(fresh["path"]))
        self.cli("submit", fresh["id"])
        self.cli("check", fresh["id"])
        self.assertEqual(self.cli("land", fresh["id"])["state"], "landed")
        self.assertTrue(Path(a["path"]).exists())

    def test_interrupted_and_failed_allocations_retain_claim_until_explicit_abandon(self):
        self.init()
        self.ticket()
        self.commit("ticket")
        result = self.cli("claim", "T-one", "--owner", "worker", failpoint="after_claim_record", ok=False)
        self.assertEqual(result.returncode, 86)
        a = self.cli("status")["attempts"][0]
        self.assertEqual(a["state"], "allocating")
        self.cli("recover", a["id"], "--confirm-stopped", ok=False)
        self.cli("claim", "T-one", "--owner", "second", ok=False)
        self.cli("abandon", a["id"], "--confirm-stopped")
        (self.root / "workspaces").write_text("operator owns this file\n")
        self.cli("claim", "T-one", "--owner", "worker", ok=False)
        second = self.cli("status")["attempts"][-1]
        self.assertEqual(second["state"], "allocating")
        self.assertEqual((self.root / "workspaces").read_text(), "operator owns this file\n")
        self.cli("claim", "T-one", "--owner", "third", ok=False)

    def test_existing_allocation_is_adopted_without_duplicate_worktree(self):
        self.init()
        self.ticket()
        self.commit("ticket")
        result = self.cli("claim", "T-one", "--owner", "worker", failpoint="after_worktree_add", ok=False)
        self.assertEqual(result.returncode, 86)
        a = self.cli("status")["attempts"][0]
        before = self.git("worktree", "list", "--porcelain")
        self.cli("recover", a["id"], ok=False)
        recovered = self.cli("recover", a["id"], "--confirm-stopped")
        self.assertEqual(recovered["state"], "working")
        self.assertEqual(self.git("worktree", "list", "--porcelain"), before)

    def test_lost_landing_acknowledgment_recovers_attributed_integration_ancestry(self):
        a = self.prepared()
        checked = self.cli("check", a["id"])
        result = self.cli("land", a["id"], failpoint="after_git_land", ok=False)
        self.assertEqual(result.returncode, 86)
        self.assertEqual(self.git("rev-parse", "HEAD"), checked["integration"])
        (self.repo / "operator.txt").write_text("after landing\n")
        head = self.commit("later operator commit")
        recovered = self.cli("recover", a["id"], "--confirm-stopped")
        self.assertEqual(recovered["state"], "landed")
        self.assertEqual(self.git("rev-parse", "HEAD"), head)
        self.assertEqual(self.git("rev-parse", "refs/collab/landed/" + a["id"]), checked["integration"])

    def test_matching_integration_without_attribution_cannot_reconcile_landing(self):
        a = self.prepared()
        checked = self.cli("check", a["id"])
        self.cli("land", a["id"], failpoint="after_land_started", ok=False)
        self.git("merge", "--ff-only", checked["integration"])
        result = self.cli("recover", a["id"], "--confirm-stopped", ok=False)
        self.assertIn("Ambiguous", result.stderr)
        self.assertEqual(self.cli("show", a["id"])["state"], "landing_started")
        self.assertEqual(self.git("rev-parse", "HEAD"), checked["integration"])

    def test_lost_reflog_never_establishes_started_landing_nonexecution(self):
        a = self.prepared()
        self.cli("check", a["id"])
        self.cli("land", a["id"], failpoint="after_land_started", ok=False)
        self.git("reflog", "expire", "--expire=all", "--all")
        result = self.cli("recover", a["id"], "--confirm-stopped", ok=False)
        self.assertIn("Ambiguous", result.stderr)
        self.assertEqual(self.cli("show", a["id"])["state"], "landing_started")

    def test_prepared_landing_with_unrelated_trunk_movement_stays_unresolved(self):
        a = self.prepared()
        self.cli("check", a["id"])
        self.cli("land", a["id"], failpoint="before_git_land", ok=False)
        (self.repo / "operator.txt").write_text("unrelated movement\n")
        head = self.commit("unrelated trunk commit")
        self.cli("recover", a["id"], "--confirm-stopped", ok=False)
        self.assertEqual(self.cli("show", a["id"])["state"], "landing_prepared")
        self.assertEqual(self.git("rev-parse", "HEAD"), head)
        self.cli("abandon", a["id"], "--confirm-stopped", ok=False)

    def test_cleanup_preserves_all_dirty_content_and_retains_history_and_old_checks(self):
        a = self.prepared()
        old = self.cli("check", a["id"])
        checked = self.cli("check", a["id"])
        self.cli("gc", a["id"], "--confirm-stopped", ok=False)
        self.cli("abandon", a["id"], "--confirm-stopped")
        self.cli("gc", a["id"], ok=False)
        worker, verification = Path(a["path"]), Path(checked["check_path"])
        ignored = verification / "build/output"
        ignored.parent.mkdir()
        ignored.write_text("ignored but valuable\n")
        self.cli("gc", a["id"], "--confirm-stopped", ok=False)
        self.assertEqual(ignored.read_text(), "ignored but valuable\n")
        self.assertTrue(worker.exists())
        ignored.unlink()
        ignored.parent.rmdir()
        (worker / "code.txt").write_text("tracked unsaved\n")
        self.cli("gc", a["id"], "--confirm-stopped", ok=False)
        self.assertEqual((worker / "code.txt").read_text(), "tracked unsaved\n")
        (worker / "code.txt").write_text("contribution\n")
        untracked = worker / "operator-notes"
        untracked.write_text("untracked valuable\n")
        self.cli("gc", a["id"], "--confirm-stopped", ok=False)
        self.assertEqual(untracked.read_text(), "untracked valuable\n")
        untracked.unlink()
        cleaned = self.cli("gc", a["id"], "--confirm-stopped")
        self.assertEqual(cleaned["removed"], [str(worker), str(verification)])
        self.assertTrue(Path(old["check_path"]).is_dir())
        self.assertTrue(Path(checked["log"]).is_file())
        self.assertEqual(self.git("rev-parse", a["branch"]), a["candidate"])
        self.assertEqual(self.git("rev-parse", "refs/collab/candidates/" + a["id"] + "/" + a["candidate"]), a["candidate"])
        self.assertEqual(self.cli("show", a["id"])["state"], "abandoned")

    def test_live_check_child_retains_lane_lock_after_cli_parent_death(self):
        started, release, done = [self.root / name for name in ("child-started", "release-child", "child-done")]
        code = ("import os,time\nfrom pathlib import Path\n"
                f"Path({str(started)!r}).write_text(str(os.getpid()))\n"
                "deadline=time.monotonic()+10\n"
                f"while not Path({str(release)!r}).exists() and time.monotonic()<deadline: time.sleep(.01)\n"
                f"Path({str(done)!r}).write_text('done')\n")
        a = self.prepared([sys.executable, "-c", code])
        process = subprocess.Popen([sys.executable, str(CLI), "--repo", str(self.repo), "check", a["id"]],
                                   env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 5
            while not started.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(started.exists(), "check child did not start")
            process.kill()
            process.communicate(timeout=5)
            result = self.cli("recover", a["id"], "--confirm-stopped", ok=False)
            self.assertIn("lane operation busy", result.stderr)
            self.assertEqual(self.cli("show", a["id"])["state"], "checking")
        finally:
            release.touch()
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            result = self.run_process([sys.executable, str(CLI), "--repo", str(self.repo),
                                       "recover", a["id"], "--confirm-stopped"])
            if result.returncode == 0:
                break
            self.assertIn("lane operation busy", result.stderr)
            time.sleep(.01)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["state"], "check_failed")
        self.assertTrue(done.exists())

    def test_backup_creates_snapshot_and_never_overwrites_existing_or_dangling_link(self):
        self.init()
        target = self.root / "snapshot.sqlite3"
        result = self.cli("backup", str(target))
        self.assertEqual(result["database_backup"], str(target))
        self.assertIn("refs, workspaces and logs", result["note"])
        before = target.read_bytes()
        self.assertTrue(before.startswith(b"SQLite format 3\0"))
        self.cli("backup", str(target), ok=False)
        self.assertEqual(target.read_bytes(), before)
        dangling, missing = self.root / "dangling.sqlite3", self.root / "missing.sqlite3"
        dangling.symlink_to(missing)
        self.cli("backup", str(dangling), ok=False)
        self.assertTrue(dangling.is_symlink())
        self.assertFalse(missing.exists())

    def test_independent_repositories_competing_for_backup_destination_have_one_winner(self):
        self.init()
        other = self.root / "other"
        other.mkdir()
        self.git("init", "-b", "main", cwd=other)
        self.git("commit", "--allow-empty", "-m", "other", cwd=other)
        self.cli("init", "--trunk", "main", "--check-json", '["true"]', repo=other)
        target = self.root / "shared-backup.sqlite3"
        processes = [subprocess.Popen([sys.executable, str(CLI), "--repo", str(repo), "backup", str(target)],
                                      env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                     for repo in (self.repo, other)]
        results = []
        for process in processes:
            out, err = process.communicate(timeout=5)
            results.append((process.returncode, out, err))
        self.assertEqual(sum(code == 0 for code, _, _ in results), 1, results)
        self.assertTrue(target.read_bytes().startswith(b"SQLite format 3\0"))

    def test_submission_requires_nonempty_linear_contribution(self):
        self.init()
        self.ticket()
        self.commit("ticket")
        a = self.claim()
        worker = Path(a["path"])
        self.cli("submit", a["id"], ok=False)
        self.git("commit", "--allow-empty", "-m", "empty worker commit", cwd=worker)
        result = self.cli("submit", a["id"], ok=False)
        self.assertIn("Empty contribution", result.stderr)
        self.git("switch", "-c", "side", cwd=worker)
        (worker / "side.txt").write_text("side\n")
        self.commit("side contribution", cwd=worker)
        self.git("switch", a["branch"], cwd=worker)
        self.contribute(a)
        self.git("merge", "--no-ff", "-m", "merge side", "side", cwd=worker)
        result = self.cli("submit", a["id"], ok=False)
        self.assertIn("linear", result.stderr)

    def test_landing_never_overwrites_ignored_operator_content(self):
        self.init()
        self.ticket()
        self.commit("ticket")
        a = self.claim()
        worker = Path(a["path"])
        (worker / "build").mkdir()
        (worker / "build/ship.txt").write_text("tracked contribution\n")
        self.git("add", "--force", "build/ship.txt", cwd=worker)
        self.commit("track formerly ignored output", cwd=worker)
        self.cli("submit", a["id"])
        self.cli("check", a["id"])
        (self.repo / "build").mkdir()
        operator_file = self.repo / "build/ship.txt"
        operator_file.write_text("ignored operator content\n")
        self.cli("land", a["id"], ok=False)
        self.assertEqual(operator_file.read_text(), "ignored operator content\n")
        self.assertEqual(self.git("rev-parse", "HEAD"), a["base"])
        self.assertEqual(self.cli("show", a["id"])["state"], "landing_started")
        self.cli("recover", a["id"], "--confirm-stopped", ok=False)

    def test_interrupted_check_before_command_is_failed_on_explicit_recovery(self):
        a = self.prepared()
        result = self.cli("check", a["id"], failpoint="before_check_command", ok=False)
        self.assertEqual(result.returncode, 86)
        interrupted = self.cli("show", a["id"])
        self.assertEqual(interrupted["state"], "checking")
        self.assertIsNone(interrupted["check_receipt"])
        self.assertFalse(Path(interrupted["log"]).exists())
        self.cli("recover", a["id"], ok=False)
        self.assertEqual(self.cli("recover", a["id"], "--confirm-stopped")["state"], "check_failed")
        self.assertEqual(self.cli("check", a["id"])["state"], "verified")

    def test_submit_refuses_skip_worktree_unsaved_content_without_clearing_flag(self):
        a = self.prepared()
        worker = Path(a["path"])
        self.git("update-index", "--skip-worktree", "code.txt", cwd=worker)
        (worker / "code.txt").write_text("hidden unsaved worker work\n")
        result = self.cli("submit", a["id"], ok=False)
        self.assertIn("hidden index", result.stderr)
        self.assertEqual((worker / "code.txt").read_text(), "hidden unsaved worker work\n")
        self.assertIn("S code.txt", self.git("ls-files", "-v", cwd=worker))

    def test_gc_preserves_assume_unchanged_unsaved_content_and_flag(self):
        a = self.prepared()
        worker = Path(a["path"])
        self.cli("abandon", a["id"], "--confirm-stopped")
        self.git("update-index", "--assume-unchanged", "code.txt", cwd=worker)
        (worker / "code.txt").write_text("hidden unsaved worker work\n")
        result = self.cli("gc", a["id"], "--confirm-stopped", ok=False)
        self.assertIn("hidden index", result.stderr)
        self.assertEqual((worker / "code.txt").read_text(), "hidden unsaved worker work\n")
        self.assertIn("h code.txt", self.git("ls-files", "-v", cwd=worker))

    def assert_hidden_check_fails(self, flag):
        code = ("import subprocess; from pathlib import Path; "
                f"subprocess.run(['git','update-index',{flag!r},'code.txt'],check=True); "
                "Path('code.txt').write_text('hidden check mutation\\n')")
        a = self.prepared([sys.executable, "-c", code])
        self.cli("check", a["id"], ok=False)
        checked = self.cli("show", a["id"])
        self.assertEqual(checked["state"], "check_failed")
        self.assertFalse(checked["check_receipt"]["clean"])
        self.assertFalse(checked["check_receipt"]["passed"])
        self.assertEqual((Path(checked["check_path"]) / "code.txt").read_text(), "hidden check mutation\n")
        self.cli("land", a["id"], ok=False)

    def test_check_cannot_verify_assume_unchanged_hidden_writes(self):
        self.assert_hidden_check_fails("--assume-unchanged")

    def test_check_cannot_verify_skip_worktree_hidden_writes(self):
        self.assert_hidden_check_fails("--skip-worktree")

    def test_land_refuses_hidden_primary_content_before_starting_git(self):
        a = self.prepared()
        self.cli("check", a["id"])
        self.git("update-index", "--assume-unchanged", "code.txt")
        (self.repo / "code.txt").write_text("hidden operator work\n")
        result = self.cli("land", a["id"], ok=False)
        self.assertIn("hidden index", result.stderr)
        self.assertEqual(self.cli("show", a["id"])["state"], "verified")
        self.assertEqual((self.repo / "code.txt").read_text(), "hidden operator work\n")

    def test_git_check_observes_integration_despite_inherited_repository_redirection(self):
        a = self.prepared(["git", "rev-parse", "HEAD"])
        redirected = dict(self.env, GIT_DIR=str(self.repo / ".git"), GIT_WORK_TREE=str(self.repo),
                          GIT_COMMON_DIR=str(self.repo / ".git"), GIT_INDEX_FILE=str(self.repo / ".git/index"))
        checked = self.cli("check", a["id"], env=redirected)
        self.assertEqual(checked["state"], "verified")
        self.assertEqual(Path(checked["log"]).read_text().splitlines()[-1], checked["integration"])
        self.assertEqual(self.git("rev-parse", "HEAD"), a["base"])

    def test_standalone_ticket_delimiters_preserve_triple_hyphens_and_crlf_bytes(self):
        self.init()
        authored = self.cli("new", "T---one", "A scalar---title")
        ticket = Path(authored["path"])
        original = (ticket.read_bytes().replace(b"deps: []\n", b"deps: []\ncontext: scalar---content\n") +
                    b"## Meaning\nKeep --- inside prose.\n\n---\nFollowing horizontal rule.\n")
        original = original.replace(b"\n", b"\r\n")
        ticket.write_bytes(original)
        self.commit("ticket with literal triple hyphens")
        self.assertEqual([t["id"] for t in self.cli("ready")], ["T---one"])
        a = self.claim("T---one")
        self.assertEqual(a["ticket_hash"], hashlib.sha256(original).hexdigest())
        self.contribute(a)
        self.cli("submit", a["id"])
        self.cli("check", a["id"])
        self.cli("land", a["id"])
        self.assertEqual(ticket.read_bytes(), original.replace(b"status: open", b"status: closed"))

    def test_completed_check_with_removed_workspace_records_failed_receipt(self):
        code = "import shutil; from pathlib import Path; shutil.rmtree(Path.cwd())"
        a = self.prepared([sys.executable, "-c", code])
        self.cli("check", a["id"], ok=False)
        current = self.cli("show", a["id"])
        self.assertEqual(current["state"], "check_failed")
        receipt = current["check_receipt"]
        self.assertEqual(receipt["exit_code"], 0)
        self.assertFalse(receipt["passed"])
        for condition in ("clean", "same_head", "identity"):
            self.assertFalse(receipt[condition])
        self.assertTrue(receipt["inspection_error"])
        self.assertEqual(receipt["subject"], current["integration"])
        self.assertEqual(receipt["trunk"], current["base"])
        self.assertFalse(Path(current["check_path"]).exists())
        self.cli("land", a["id"], ok=False)

    def test_new_rejects_duplicate_and_self_dependencies_without_dirtying_operator(self):
        self.init()
        for deps in (["T-new"], ["T-parent", "T-parent"]):
            with self.subTest(deps=deps):
                result = self.cli("new", "T-new", "Invalid dependencies", "--deps", *deps, ok=False)
                self.assertIn("dependency", result.stderr)
                self.assertEqual(self.git("status", "--porcelain=v1"), "")
                self.assertFalse((self.repo / ".tickets").exists())


if __name__ == "__main__":
    unittest.main()
