# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Task-specific Harbor agent for the Terminal-Bench headless-terminal smoke.

This module is intentionally imported only by Harbor's custom agent loader. It
keeps the Harbor dependency out of the normal scaffold-to-policy Python surface.
"""

from __future__ import annotations

from textwrap import dedent

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


_HEADLESS_TERMINAL_IMPL = dedent(
    """
    import subprocess
    import time
    import uuid

    from base_terminal import BaseTerminal


    class HeadlessTerminal(BaseTerminal):
        def __init__(self):
            self._session_id = str(uuid.uuid4())
            subprocess.run(
                ["tmux", "new-session", "-d", "-s", self._session_id],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        def send_keystrokes(self, keystrokes: str, wait_sec: float = 0.0) -> None:
            subprocess.run(
                ["tmux", "send-keys", "-t", self._session_id, keystrokes],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(wait_sec)
    """
).strip() + "\n"


class HeadlessTerminalScriptAgent(BaseAgent):
    """Bounded Harbor custom agent for one pinned Terminal-Bench task."""

    @staticmethod
    def name() -> str:
        return "torchtitan-headless-terminal-script"

    def version(self) -> str:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        result = await environment.exec(
            command="command -v tmux >/dev/null && test -f /app/base_terminal.py",
            timeout_sec=30,
        )
        if result.return_code != 0:
            raise RuntimeError(
                "headless-terminal task container is missing tmux or "
                f"/app/base_terminal.py: {result.stderr or result.stdout or ''}"
            )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        command = "\n".join(
            [
                "cat > /app/headless_terminal.py <<'PY'",
                _HEADLESS_TERMINAL_IMPL,
                "PY",
                "python -m py_compile /app/headless_terminal.py",
            ]
        )
        result = await environment.exec(command=command, timeout_sec=30)
        context.metadata = {
            "agent_kind": "task_specific_scripted_policy",
            "task_id": "headless-terminal",
            "instruction_chars": len(instruction),
            "write_return_code": result.return_code,
            "stdout_tail": (result.stdout or "")[-1000:],
            "stderr_tail": (result.stderr or "")[-1000:],
        }
        if result.return_code != 0:
            raise RuntimeError(
                "failed to write /app/headless_terminal.py: "
                f"{result.stderr or result.stdout or ''}"
            )
