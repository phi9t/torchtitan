# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Task-specific Harbor agent for the Terminal-Bench headless-terminal smoke.

This module is intentionally imported only by Harbor's custom agent loader. It
keeps the Harbor dependency out of the normal scaffold-to-policy Python surface.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import subprocess
import tempfile
from textwrap import dedent

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


_FENCED_CODE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

_HEADLESS_TERMINAL_IMPL = (
    dedent(
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
    ).strip()
    + "\n"
)

_MODEL_PROMPT = dedent(
    """
    You are solving a Terminal-Bench task. Write the complete contents of
    /app/headless_terminal.py.

    Requirements:
    - Define class HeadlessTerminal(BaseTerminal).
    - Import BaseTerminal from base_terminal.
    - The class must expose send_keystrokes(self, keystrokes: str,
      wait_sec: float = 0.0) -> None.
    - It must support interactive bash state across calls.
    - It must support modifier/control characters such as Ctrl-C.
    - It must source normal interactive shell startup files.
    - It must support background commands continuing after send_keystrokes returns.
    - Use only Python standard library and tools expected in a Linux terminal task.

    Return only Python code. Do not include Markdown, prose, or tests.
    """
).strip()


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


def _extract_python_code(text: str) -> str:
    matches = _FENCED_CODE_RE.findall(text.strip())
    if matches:
        return matches[-1].strip() + "\n"
    candidate = text.strip()
    lines = candidate.splitlines()
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(("from ", "import ", "class ", "@")):
            candidate = "\n".join(lines[index:]).strip()
            break
    if "from base_terminal import BaseTerminal" in candidate:
        return candidate + "\n"
    marker = "class HeadlessTerminal"
    index = candidate.find(marker)
    if index >= 0:
        return "from base_terminal import BaseTerminal\n\n" + candidate[index:] + "\n"
    return candidate + "\n"


def _generate_headless_terminal_code() -> tuple[str, str]:
    model_path = os.environ.get(
        "SCAFFOLD_TO_POLICY_HARBOR_MODEL", "./assets/hf/Qwen3-1.7B"
    )
    gpu_memory_utilization = float(
        os.environ.get("SCAFFOLD_TO_POLICY_HARBOR_GPU_MEMORY_UTILIZATION", "0.05")
    )
    max_model_len = int(
        os.environ.get("SCAFFOLD_TO_POLICY_HARBOR_MAX_MODEL_LEN", "2048")
    )
    max_tokens = int(os.environ.get("SCAFFOLD_TO_POLICY_HARBOR_MAX_TOKENS", "768"))
    temperature = float(os.environ.get("SCAFFOLD_TO_POLICY_HARBOR_TEMPERATURE", "0.2"))
    payload = {
        "prompt": _MODEL_PROMPT,
        "model_path": model_path,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "attention_backend": os.environ.get(
            "SCAFFOLD_TO_POLICY_HARBOR_ATTENTION_BACKEND", "TRITON_ATTN"
        ),
    }
    script = dedent(
        """
        import json
        import os
        import re
        import sys
        from pathlib import Path

        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        FENCED_CODE_RE = re.compile(
            r"```(?:python|py)?\\s*(.*?)```", re.DOTALL | re.IGNORECASE
        )

        def extract_python_code(text: str) -> str:
            matches = FENCED_CODE_RE.findall(text.strip())
            if matches:
                return matches[-1].strip() + "\\n"
            candidate = text.strip()
            if "from base_terminal import BaseTerminal" in candidate:
                return candidate + "\\n"
            marker = "class HeadlessTerminal"
            index = candidate.find(marker)
            if index >= 0:
                return (
                    "from base_terminal import BaseTerminal\\n\\n"
                    + candidate[index:]
                    + "\\n"
                )
            return candidate + "\\n"

        payload = json.loads(sys.stdin.read())
        tokenizer = AutoTokenizer.from_pretrained(payload["model_path"])
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        "You solve Python programming tasks. Return only Python code, "
                        "with no Markdown. Preserve the requested class name, method "
                        "name, signature, side effects, and error behavior."
                    ),
                },
                {"role": "user", "content": payload["prompt"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        llm = LLM(
            model=payload["model_path"],
            attention_backend=payload["attention_backend"],
            max_model_len=payload["max_model_len"],
            gpu_memory_utilization=payload["gpu_memory_utilization"],
            enable_flashinfer_autotune=False,
            disable_log_stats=True,
        )
        params = SamplingParams(
            n=1,
            temperature=payload["temperature"],
            top_p=0.95,
            max_tokens=payload["max_tokens"],
        )
        outputs = llm.generate([prompt], params)
        raw_text = outputs[0].outputs[0].text
        result = json.dumps({"code": extract_python_code(raw_text), "raw_text": raw_text})
        output_path = os.environ.get("SCAFFOLD_TO_POLICY_HARBOR_GENERATION_OUTPUT")
        if output_path:
            Path(output_path).write_text(result)
        else:
            print(result)
        """
    )
    env = os.environ.copy()
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    with tempfile.NamedTemporaryFile(
        "r", suffix=".json", delete=False
    ) as output_handle:
        output_path = output_handle.name
    try:
        env["SCAFFOLD_TO_POLICY_HARBOR_GENERATION_OUTPUT"] = output_path
        completed = subprocess.run(
            ["/usr/bin/python", "-c", script],
            env=env,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
            timeout=int(
                os.environ.get("SCAFFOLD_TO_POLICY_HARBOR_GENERATE_TIMEOUT", "180")
            ),
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "rootfs vLLM generation failed: "
                f"{completed.stderr[-4000:] or completed.stdout[-4000:]}"
            )
        with open(output_path) as handle:
            result = json.loads(handle.read())
        return str(result["code"]), str(result["raw_text"])
    finally:
        try:
            os.unlink(output_path)
        except FileNotFoundError:
            pass


class HeadlessTerminalQwenAgent(BaseAgent):
    """Qwen3/vLLM Harbor agent for the pinned headless-terminal task."""

    @staticmethod
    def name() -> str:
        return "torchtitan-headless-terminal-qwen"

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
        code, raw_text = await asyncio.to_thread(_generate_headless_terminal_code)
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
            handle.write(code)
            local_path = handle.name
        try:
            compile_result = subprocess.run(
                ["/usr/bin/python", "-m", "py_compile", local_path],
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
            )
            encoded_code = base64.b64encode(code.encode()).decode()
            result = await environment.exec(
                command=(
                    f"python - <<'PY'\n"
                    f"import base64\n"
                    f"from pathlib import Path\n"
                    f"data = base64.b64decode({encoded_code!r}).decode()\n"
                    f"Path('/app/headless_terminal.py').write_text(data)\n"
                    f"PY\n"
                ),
                timeout_sec=30,
            )
            context.metadata = {
                "agent_kind": "qwen_vllm_policy",
                "task_id": "headless-terminal",
                "instruction_chars": len(instruction),
                "local_compile_return_code": compile_result.returncode,
                "local_compile_stderr_tail": compile_result.stderr[-1000:],
                "container_write_return_code": result.return_code,
                "raw_text_tail": raw_text[-2000:],
                "code_tail": code[-2000:],
                "stdout_tail": (result.stdout or "")[-1000:],
                "stderr_tail": (result.stderr or "")[-1000:],
            }
            if result.return_code != 0:
                raise RuntimeError(
                    "failed to write model-generated /app/headless_terminal.py: "
                    f"{result.stderr or result.stdout or ''}"
                )
        finally:
            try:
                os.unlink(local_path)
            except FileNotFoundError:
                pass
