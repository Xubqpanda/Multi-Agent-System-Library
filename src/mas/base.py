# src/mas/base.py
"""
MAS 层核心抽象。

MetaMAS：所有 MAS 框架的统一接口，只暴露 build_system 和 run_task 两个方法。
Agent：单个 agent 的通用封装。
Env：环境交互接口。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Iterable

from src.llm import Message
from src.mas.reasoning import ReasoningBase, ReasoningConfig
from src.memory.base import MASMemoryBase


# ─── Agent ────────────────────────────────────────────────────────────────────

class Agent:
    """单个 agent 的通用封装。"""

    def __init__(
        self,
        name: str,
        role: str,
        system_instruction: str,
        reasoning_module: ReasoningBase,
        memory_module=None,
    ):
        if reasoning_module is None:
            raise ValueError("reasoning_module must not be None.")
        self.name = name
        self.profile = role
        self.system_instruction = system_instruction
        self.reasoning = reasoning_module
        self.memory = memory_module
        self.total_system_instruction = system_instruction

    def add_task_instruction(self, task_instruction: str) -> str:
        self.total_system_instruction = self.system_instruction + "\n" + task_instruction
        return self.total_system_instruction

    def response(self, user_prompt: str, reason_config: ReasoningConfig) -> str:
        messages = [
            Message("system", self.total_system_instruction),
            Message("user", user_prompt),
        ]
        return self.reasoning(messages, reason_config)


# ─── Env ──────────────────────────────────────────────────────────────────────

class Env:
    """环境交互接口，子类根据具体任务重写。"""

    def __init__(self):
        pass

    def set_env(self, configs: dict) -> None:
        pass

    def reset(self) -> None:
        pass

    def step(self, action: str) -> tuple[str, float, bool]:
        """执行 action，返回 (observation, reward, done)。"""
        raise NotImplementedError

    def feedback(self) -> tuple[float, bool, str]:
        """任务结束后的最终反馈，返回 (final_reward, done, feedback_str)。"""
        raise NotImplementedError

    def process_action(self, action: str) -> str:
        """对 action 进行预处理（如格式化）。默认返回原始 action。"""
        return action


# ─── MetaMAS ──────────────────────────────────────────────────────────────────

@dataclass
class MetaMAS(ABC):
    """
    所有 MAS 框架的统一抽象基类。

    核心契约：
      - build_system：注入 reasoning、memory、env 和框架超参，完成框架内部初始化。
      - run_task：接受 task_config dict，返回 (reward, success)。

    框架内部的 agents、workflow、拓扑等实现完全自由，对外只暴露这两个方法。
    """
    agents_team: Dict[str, Agent] = field(default_factory=dict)
    env: Optional[Env] = None
    meta_memory: Optional[MASMemoryBase] = None

    # ── 工具方法（供子类使用） ─────────────────────────────────────────────────

    def hire(self, agents: Iterable[Agent]) -> None:
        for agent in agents:
            if agent.name not in self.agents_team:
                self.agents_team[agent.name] = agent
            else:
                print(f"[MetaMAS] Agent '{agent.name}' already in team, skipped.")

    def set_env(self, env: Env) -> None:
        self.env = env

    def get_agent(self, agent_name: str) -> Optional[Agent]:
        return self.agents_team.get(agent_name)

    # ── 抽象接口 ───────────────────────────────────────────────────────────────

    @abstractmethod
    def build_system(
        self,
        reasoning: ReasoningBase,
        memory: MASMemoryBase,
        env: Env,
        config: dict,
    ) -> None:
        """完成框架内部组件的初始化与连接。"""
        pass

    @abstractmethod
    def run_task(self, task_config: dict) -> tuple[float, bool]:
        """
        执行单个任务。

        Args:
            task_config: 任务配置字典，至少包含 task_main 和 task_description。

        Returns:
            (reward, success): 最终奖励分数和是否成功。
        """
        pass