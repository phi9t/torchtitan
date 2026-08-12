# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Deterministic tau2 agent registration for rootfs execution probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


AGENT_NAME = "torchtitan_mock_oracle_agent"
NOOP_AGENT_NAME = "torchtitan_noop_agent"
USER_NAME = "torchtitan_static_user"


@dataclass
class _ProbeAgentState:
    next_action_index: int = 0
    done_sent: bool = False


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
    if USER_NAME not in registry.get_users():
        registry.register_user(TorchTitanStaticUser, USER_NAME)


register_tau2_probe_agent()
