# src/memory/base.py
"""
MASMemoryBase：所有 memory 方法的统一抽象基类。

设计原则：
  - inside-trial memory（任务进行中）：init_task_context / add_agent_node / move_memory_state / save_task_context
  - cross-trial memory（跨任务）：add_memory / retrieve_memory / update_memory / backward
  子类只需覆盖 cross-trial 部分即可实现新的 memory 方法。
"""
import os
from dataclasses import dataclass
from abc import ABC

from src.common import AgentMessage, MASMessage, StorageNameSpace
from src.llm import LLMCallable
from src.utils import EmbeddingFunc


@dataclass
class MASMemoryBase(StorageNameSpace, ABC):
    """
    MAS Memory 抽象基类。

    Attributes:
        llm_model (LLMCallable): 用于生成摘要、洞见等的 LLM 调用接口。
        embedding_func (EmbeddingFunc): 向量化函数，用于语义检索。
    """
    llm_model: LLMCallable
    embedding_func: EmbeddingFunc

    def __post_init__(self):
        self.persist_dir: str = os.path.join(self.global_config["working_dir"], self.namespace)
        os.makedirs(self.persist_dir, exist_ok=True)
        self.current_task_context: MASMessage = None

    # ── Inside-trial memory（任务内，所有子类共用） ────────────────────────────

    def init_task_context(
        self,
        task_main: str,
        task_description: str = None,
    ) -> MASMessage:
        """初始化当前任务的 memory 上下文。每个任务开始时调用一次。"""
        self.current_task_context = MASMessage(
            task_main=task_main,
            task_description=task_description,
        )
        return self.current_task_context

    def add_agent_node(
        self,
        agent_message: AgentMessage,
        upstream_agent_ids: list[str],
    ) -> str:
        """将 agent 的一次响应记录到当前状态图中。"""
        return self.current_task_context.add_message_to_current_state(
            agent_message, upstream_agent_ids
        )

    def move_memory_state(self, action: str, observation: str, **kwargs) -> None:
        """推进状态链到下一个状态（对应环境执行一步之后）。"""
        self.current_task_context.move_state(action, observation, **kwargs)

    def save_task_context(self, label: bool, feedback: str = None) -> MASMessage:
        """任务结束时调用，打标签并持久化到 cross-trial memory。"""
        if self.current_task_context is None:
            raise RuntimeError("current_task_context is None, call init_task_context first.")
        self.current_task_context.label = label
        if feedback is not None:
            self.current_task_context.task_description += f"\n- Environment feedback\n{feedback}\n"
        self.add_memory(self.current_task_context)
        return self.current_task_context

    def summarize(self, **kwargs) -> str:
        """返回当前任务上下文的文本摘要（description + trajectory）。"""
        ctx = self.current_task_context
        return (ctx.task_description or "") + (ctx.task_trajectory or "")

    # ── Cross-trial memory（子类覆盖以实现不同方法） ──────────────────────────

    def add_memory(self, mas_message: MASMessage) -> None:
        """将已完成任务的 MASMessage 存入 memory。默认空实现（no-memory baseline）。"""
        pass

    def retrieve_memory(
        self,
        query_task: str,
        successful_topk: int = 1,
        failed_topk: int = 1,
        **kwargs,
    ) -> tuple[list[MASMessage], list[MASMessage], list]:
        """
        检索相关历史记忆。

        Returns:
            (successful_trajectories, failed_trajectories, insights)
        """
        return [], [], []

    def update_memory(self, query: str, **kwargs) -> None:
        """在线更新 memory（如 insight 更新）。默认空实现。"""
        pass

    def backward(self, reward: bool, **kwargs) -> None:
        """基于任务结果反向更新 memory（如强化学习风格的更新）。默认空实现。"""
        pass