import json
import subprocess
from pathlib import Path

from tools import active_work


def _git_init(path: Path) -> None:
    path.mkdir(parents=True)
    subprocess.run(
        ("git", "init", "-b", "main"),
        cwd=path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_load_workspace_config_maps_ultron_mainline_manifest_fields(tmp_path):
    workspace = tmp_path / "workspace"
    checkout = workspace / "torchtitan"
    _git_init(checkout)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": "2026.08.15.2",
                "source_repository": "torchtitan",
                "repositories": [
                    {
                        "name": "torchtitan",
                        "upstream_mainline": "main",
                        "ultron_mainline": "ultron/mainline",
                        "ultron_mainline_bootstrap": "d9d58bf2d3fe5f0f4434cf9aa943a867d8c551f3",
                        "instruction_files": [".claude/CLAUDE.md"],
                        "install_local_tracker": True,
                    }
                ],
            }
        )
    )

    config = active_work.load_workspace_config(
        manifest,
        workspace,
        repository_roots={"torchtitan": checkout},
        enforce_canonical_topology=False,
    )

    assert config.version == "2026.08.15.2"
    titan = config.repositories[0]
    assert titan.mainline == "main"
    assert titan.kata_project == "torchtitan"
    assert titan.legacy_tracker == "markdown"
