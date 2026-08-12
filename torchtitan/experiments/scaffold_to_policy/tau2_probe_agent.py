# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Deterministic tau2 agent registration for rootfs execution probes."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any


AGENT_NAME = "torchtitan_mock_oracle_agent"
NOOP_AGENT_NAME = "torchtitan_noop_agent"
QWEN_AGENT_NAME = "torchtitan_qwen_agent"
USER_NAME = "torchtitan_static_user"


@dataclass
class _ProbeAgentState:
    next_action_index: int = 0
    done_sent: bool = False
    turn_index: int = 0


@dataclass
class _StaticUserState:
    initial_message_sent: bool = False


def register_tau2_probe_agent() -> None:
    from tau2.agent.base_agent import HalfDuplexAgent
    from tau2.agent.llm_agent import LLMSoloAgent
    from tau2.data_model.message import AssistantMessage, ToolCall
    from tau2.data_model.message import UserMessage
    from tau2.data_model.tasks import Task
    from tau2.environment.tool import Tool
    from tau2.registry import registry
    from tau2.user.user_simulator_base import STOP, HalfDuplexUser

    if AGENT_NAME in registry.get_agents():
        agent_registered = True
    else:
        agent_registered = False
    noop_agent_registered = NOOP_AGENT_NAME in registry.get_agents()
    qwen_agent_registered = QWEN_AGENT_NAME in registry.get_agents()

    class TorchTitanMockOracleAgent(HalfDuplexAgent[_ProbeAgentState]):
        STOP_FUNCTION_NAME = LLMSoloAgent.STOP_FUNCTION_NAME
        STOP_TOKEN = LLMSoloAgent.STOP_TOKEN

        def __init__(
            self,
            tools: list[Tool],
            domain_policy: str,
            task: Task,
            **_: Any,
        ) -> None:
            super().__init__(tools=tools, domain_policy=domain_policy)
            self.task = task
            self.assistant_actions = [
                action
                for action in (task.evaluation_criteria.actions or [])
                if action.requestor == "assistant"
            ]

        @classmethod
        def check_valid_task(cls, task: Task) -> bool:
            if task.evaluation_criteria is None:
                return False
            return any(
                action.requestor == "assistant"
                for action in (task.evaluation_criteria.actions or [])
            )

        def get_init_state(self, message_history=None) -> _ProbeAgentState:
            return _ProbeAgentState()

        @classmethod
        def is_stop(cls, message: AssistantMessage) -> bool:
            return message.content is not None and cls.STOP_TOKEN in message.content

        def generate_next_message(
            self,
            message,
            state: _ProbeAgentState,
        ) -> tuple[AssistantMessage, _ProbeAgentState]:
            if state.next_action_index < len(self.assistant_actions):
                action = self.assistant_actions[state.next_action_index]
                state.next_action_index += 1
                tool_call = ToolCall(
                    id=f"torchtitan_call_{state.next_action_index}",
                    name=action.name,
                    arguments=dict(action.arguments or {}),
                    requestor="assistant",
                )
                return AssistantMessage.text(content=None, tool_calls=[tool_call]), state
            state.done_sent = True
            return AssistantMessage.text(content=self.STOP_TOKEN), state

    class TorchTitanNoopAgent(HalfDuplexAgent[_ProbeAgentState]):
        STOP_TOKEN = LLMSoloAgent.STOP_TOKEN

        def __init__(
            self,
            tools: list[Tool],
            domain_policy: str,
            **_: Any,
        ) -> None:
            super().__init__(tools=tools, domain_policy=domain_policy)

        def get_init_state(self, message_history=None) -> _ProbeAgentState:
            return _ProbeAgentState(done_sent=True)

        @classmethod
        def is_stop(cls, message: AssistantMessage) -> bool:
            return message.content is not None and cls.STOP_TOKEN in message.content

        def generate_next_message(
            self,
            message,
            state: _ProbeAgentState,
        ) -> tuple[AssistantMessage, _ProbeAgentState]:
            state.done_sent = True
            return AssistantMessage.text(content=self.STOP_TOKEN), state

    class TorchTitanQwenAgent(HalfDuplexAgent[_ProbeAgentState]):
        STOP_TOKEN = LLMSoloAgent.STOP_TOKEN

        def __init__(
            self,
            tools: list[Tool],
            domain_policy: str,
            **_: Any,
        ) -> None:
            super().__init__(tools=tools, domain_policy=domain_policy)

        def get_init_state(self, message_history=None) -> _ProbeAgentState:
            return _ProbeAgentState()

        @classmethod
        def is_stop(cls, message: AssistantMessage) -> bool:
            return message.content is not None and cls.STOP_TOKEN in message.content

        def generate_next_message(
            self,
            message,
            state: _ProbeAgentState,
        ) -> tuple[AssistantMessage, _ProbeAgentState]:
            state.turn_index += 1
            decision = _generate_qwen_tau2_action(
                domain_policy=self.domain_policy,
                tools=self.tools,
                latest_message=message,
                turn_index=state.turn_index,
            )
            if decision.get("type") == "tool":
                tool_name = str(decision.get("name", ""))
                if tool_name in {tool.name for tool in self.tools}:
                    arguments = decision.get("arguments")
                    if isinstance(arguments, dict):
                        tool_call = ToolCall(
                            id=f"torchtitan_qwen_call_{state.turn_index}",
                            name=tool_name,
                            arguments=arguments,
                            requestor="assistant",
                        )
                        return AssistantMessage.text(content=None, tool_calls=[tool_call]), state
            state.done_sent = True
            content = str(decision.get("content") or self.STOP_TOKEN)
            if self.STOP_TOKEN not in content:
                content = f"{content}\n{self.STOP_TOKEN}"
            return AssistantMessage.text(content=content), state

    class TorchTitanStaticUser(HalfDuplexUser[_StaticUserState]):
        def __init__(
            self,
            instructions: str | None = None,
            tools: list[Tool] | None = None,
            **_: Any,
        ) -> None:
            super().__init__(instructions=instructions, tools=tools)

        def get_init_state(self, message_history=None) -> _StaticUserState:
            return _StaticUserState()

        @classmethod
        def is_stop(cls, message: UserMessage) -> bool:
            return message.content is not None and STOP in message.content

        def generate_next_message(
            self,
            message,
            state: _StaticUserState,
        ) -> tuple[UserMessage, _StaticUserState]:
            if not state.initial_message_sent:
                state.initial_message_sent = True
                return UserMessage.text(content=self.instructions or ""), state
            return UserMessage.text(content=STOP), state

    def create_torchtitan_mock_oracle_agent(
        tools,
        domain_policy,
        **kwargs,
    ) -> TorchTitanMockOracleAgent:
        return TorchTitanMockOracleAgent(
            tools=tools,
            domain_policy=domain_policy,
            task=kwargs["task"],
        )

    def create_torchtitan_noop_agent(
        tools,
        domain_policy,
        **kwargs,
    ) -> TorchTitanNoopAgent:
        return TorchTitanNoopAgent(
            tools=tools,
            domain_policy=domain_policy,
        )

    def create_torchtitan_qwen_agent(
        tools,
        domain_policy,
        **kwargs,
    ) -> TorchTitanQwenAgent:
        return TorchTitanQwenAgent(
            tools=tools,
            domain_policy=domain_policy,
        )

    if not agent_registered:
        registry.register_agent_factory(
            create_torchtitan_mock_oracle_agent,
            AGENT_NAME,
            task_filter=TorchTitanMockOracleAgent.check_valid_task,
            metadata={"solo_mode": False},
        )
    if not noop_agent_registered:
        registry.register_agent_factory(
            create_torchtitan_noop_agent,
            NOOP_AGENT_NAME,
            metadata={"solo_mode": False},
        )
    if not qwen_agent_registered:
        registry.register_agent_factory(
            create_torchtitan_qwen_agent,
            QWEN_AGENT_NAME,
            metadata={"solo_mode": False},
        )
    if USER_NAME not in registry.get_users():
        registry.register_user(TorchTitanStaticUser, USER_NAME)


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_model_dump(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _model_dump(item) for key, item in value.items()}
    return str(value)


def _truncate_text(value: Any, *, limit: int = 600) -> Any:
    if not isinstance(value, str) or len(value) <= limit:
        return value
    return value[:limit] + "...[truncated]"


def _compact_message(value: Any) -> Any:
    if not isinstance(value, dict):
        return _truncate_text(value)
    compact: dict[str, Any] = {}
    for key in (
        "role",
        "content",
        "tool_calls",
        "name",
        "arguments",
        "requestor",
        "error",
    ):
        if key in value and value[key] is not None:
            compact[key] = _truncate_text(value[key])
    return compact


def _compact_recent_messages(value: Any, *, limit: int = 6) -> Any:
    if isinstance(value, list):
        return [_compact_message(item) for item in value[-limit:]]
    return _compact_message(value)


def _compact_tool(value: Any) -> Any:
    dumped = _model_dump(value)
    if isinstance(dumped, dict):
        compact = {
            key: _truncate_text(dumped[key], limit=300)
            for key in ("name", "description")
            if key in dumped
        }
        for key in ("parameters", "schema", "args_schema"):
            if key in dumped:
                compact[key] = dumped[key]
                break
        return compact
    return _truncate_text(dumped, limit=300)


def _extract_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"type": "stop", "content": "No JSON action was emitted."}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"type": "stop", "content": "Invalid JSON action was emitted."}
    if isinstance(payload, dict):
        return payload
    return {"type": "stop", "content": "JSON action was not an object."}


def _generate_qwen_tau2_action(
    *,
    domain_policy: str,
    tools: list[Any],
    latest_message: Any,
    turn_index: int,
) -> dict[str, Any]:
    model_path = os.environ.get("SCAFFOLD_TO_POLICY_TAU2_MODEL", "./assets/hf/Qwen3-1.7B")
    dumped_latest_message = _model_dump(latest_message)
    recent_messages = _compact_recent_messages(dumped_latest_message)
    payload = {
        "domain_policy": domain_policy,
        "tools": [_compact_tool(tool) for tool in tools],
        "latest_message": recent_messages[-1] if isinstance(recent_messages, list) else recent_messages,
        "recent_messages": recent_messages,
        "turn_index": turn_index,
        "model_path": model_path,
        "gpu_memory_utilization": float(
            os.environ.get("SCAFFOLD_TO_POLICY_TAU2_GPU_MEMORY_UTILIZATION", "0.05")
        ),
        "max_model_len": int(os.environ.get("SCAFFOLD_TO_POLICY_TAU2_MAX_MODEL_LEN", "2048")),
        "max_tokens": int(os.environ.get("SCAFFOLD_TO_POLICY_TAU2_MAX_TOKENS", "512")),
        "temperature": float(os.environ.get("SCAFFOLD_TO_POLICY_TAU2_TEMPERATURE", "0.2")),
        "attention_backend": os.environ.get(
            "SCAFFOLD_TO_POLICY_TAU2_ATTENTION_BACKEND", "TRITON_ATTN"
        ),
    }
    script = r'''
import json
import os
import sys
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

payload = json.loads(sys.stdin.read())
system = (
    "You are a tau2-bench agent. Choose exactly one next action. "
    "Return only JSON. To call a tool, return "
    "{\"type\":\"tool\",\"name\":\"tool_name\",\"arguments\":{...}}. "
    "To stop or answer the user, return {\"type\":\"stop\",\"content\":\"...\"}. "
    "Use only listed tools and obey the domain policy. "
    "If the most recent tool result reports success, do not call the tool again; "
    "confirm completion with a stop action."
)
user = json.dumps(
    {
        "domain_policy": payload["domain_policy"],
        "tools": payload["tools"],
        "latest_message": payload["latest_message"],
        "recent_messages": payload["recent_messages"],
        "turn_index": payload["turn_index"],
    },
    indent=2,
    sort_keys=True,
)
tokenizer = AutoTokenizer.from_pretrained(payload["model_path"])
prompt = tokenizer.apply_chat_template(
    [{"role": "system", "content": system}, {"role": "user", "content": user}],
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
result = json.dumps({"raw_text": raw_text})
output_path = os.environ.get("SCAFFOLD_TO_POLICY_TAU2_GENERATION_OUTPUT")
if output_path:
    Path(output_path).write_text(result)
else:
    print(result)
'''
    env = os.environ.copy()
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as output_handle:
        output_path = output_handle.name
    try:
        env["SCAFFOLD_TO_POLICY_TAU2_GENERATION_OUTPUT"] = output_path
        completed = subprocess.run(
            ["/usr/bin/python", "-c", script],
            env=env,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
            timeout=int(os.environ.get("SCAFFOLD_TO_POLICY_TAU2_GENERATE_TIMEOUT", "180")),
        )
        if completed.returncode != 0:
            return {
                "type": "stop",
                "content": "Local Qwen generation failed: "
                + (completed.stderr[-1000:] or completed.stdout[-1000:]),
            }
        with open(output_path) as handle:
            raw = json.loads(handle.read()).get("raw_text", "")
        return _extract_json_object(str(raw))
    finally:
        try:
            os.unlink(output_path)
        except FileNotFoundError:
            pass


register_tau2_probe_agent()
