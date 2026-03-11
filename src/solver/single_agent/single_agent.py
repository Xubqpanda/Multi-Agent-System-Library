# src/solver/single_agent/single_agent.py
"""
SingleAgentSolver：单 agent solver baseline。

继承 MetaSolver，run_task 主循环精简为：

    init_working_memory(task_main, task_description, context_hint)
        └── memory 内部决定是否预加载 experiential memory（第一、二层）

    for i in range(max_trials):
        user_prompt = retrieve_working_memory()   # memory 决定 prompt 内容
        answer      = reasoning(messages)
        add_working_memory(AgentMessage)
        observation, reward, done = env.step(answer)
        add_working_memory((answer, observation), reward=reward)
        if done: break

    final_reward, final_done, final_feedback = env.feedback()
    add_experiential_memory(label=final_done, feedback=final_feedback)

设计原则：
  - solver 只负责驱动推理和 env 交互，完全不感知 prompt 构造细节。
  - retrieve_experiential_memory 是 agent 的主动按需调用接口（SkillMem 第三层），
    solver 在此不调用，由继承此 solver 的 SkillMem-aware solver 决定何时触发。
  - 检索超参（topk、threshold 等）属于各 memory 方法的内部配置，solver 不传递。
"""

from typing import Optional
from dataclasses import dataclass

from src.envs.base      import Env
from src.solver.base    import MetaSolver
from src.reasoning      import ReasoningBase, ReasoningConfig
from src.memory.base    import MemoryBase
from src.common.message import AgentMessage
from src.llm            import Message, token_tracker

SINGLE_AGENT_SYSTEM_PROMPT = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answer choice}\n"
    "Answer: {your chosen answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)


@dataclass
class SingleAgentSolver(MetaSolver):
    """单 agent solver baseline，继承 MetaSolver。"""

    def __post_init__(self):
        self.observers        = []
        self.reasoning_config = ReasoningConfig(temperature=0)

    # ── build_system ──────────────────────────────────────────────────────────

    def build_system(
        self,
        reasoning: ReasoningBase,
        solver_memory: MemoryBase,
        env: Env,
        config: dict,
    ) -> None:
        """
        注入推理模块、memory、环境，完成初始化。

        config 目前只支持一个字段：
          system_prompt (str) : 覆盖默认 system prompt（可选）
        """
        if not isinstance(reasoning, ReasoningBase):
            raise TypeError("reasoning must be an instance of ReasoningBase")
        if not isinstance(solver_memory, MemoryBase):
            raise TypeError("solver_memory must be an instance of MemoryBase")
        if not isinstance(env, Env):
            raise TypeError("env must be an instance of Env")

        self._system_prompt: str = config.get("system_prompt", SINGLE_AGENT_SYSTEM_PROMPT)
        self._reasoning          = reasoning
        self.meta_memory         = solver_memory
        self.set_env(env)

    # ── run_task ──────────────────────────────────────────────────────────────

    def run_task(self, task_config: dict) -> tuple[float, bool]:
        """
        执行单个任务。

        task_config 字段：
          task_main        (str)  : 题目核心内容，必填
          task_description (str)  : 完整任务描述，默认同 task_main
          context_hint     (dict) : 可选任务元信息，透传给 memory
                                    特殊 key：image_b64 / image_media_type（多模态）
          max_trials       (int)  : 最大交互步数；
                                    未指定时优先读 env.max_trials，默认 1（QA 场景）
        """
        if task_config.get("task_main") is None:
            raise ValueError("Missing required key 'task_main' in task_config")

        task_main:        str  = task_config["task_main"]
        task_description: str  = task_config.get("task_description", task_main)
        context_hint:     dict = task_config.get("context_hint", {})
        max_trials:       int  = task_config.get(
            "max_trials", getattr(self.env, "max_trials", 1)
        )

        env = self.env
        env.reset()

        # ── 初始化 working memory ──────────────────────────────────────────
        # memory 内部决定是否预加载 experiential memory（及加载哪几层）
        self.meta_memory.init_working_memory(
            task_main=task_main,
            task_description=task_description,
            context_hint=context_hint,
        )

        # ── 多模态图片（与 memory 无关，在 solver 层处理）─────────────────
        image_b64:        Optional[str] = context_hint.get("image_b64")
        image_media_type: str           = context_hint.get("image_media_type", "image/jpeg")

        # ── 主循环 ─────────────────────────────────────────────────────────
        for i in range(max_trials):

            # memory 决定返回什么 prompt（EmptyMemory 只返回题目，其他可能含经验）
            user_prompt: str = self.meta_memory.retrieve_working_memory()
            self.notify_observers(user_prompt)

            # 构造 messages：多模态时 user content 变为 list
            if image_b64:
                user_content = [
                    {"type": "input_text",  "text": user_prompt},
                    {"type": "input_image", "image_url": {
                        "url": f"data:{image_media_type};base64,{image_b64}"
                    }},
                ]
            else:
                user_content = user_prompt

            messages = [
                Message("system", self._system_prompt),
                Message("user",   user_content),
            ]

            answer: str = self._reasoning(messages, self.reasoning_config)
            self.notify_observers(f"Step {i+1} Answer: {answer}")

            self.meta_memory.add_working_memory(
                AgentMessage(
                    agent_name="solver",
                    user_instruction=user_prompt,
                    message=answer,
                ),
                upstream_ids=[],
            )

            observation, reward, done = env.step(answer)
            self.notify_observers(f"Act {i+1}: {answer}\nObs {i+1}: {observation}")

            self.meta_memory.add_working_memory(
                (answer, observation),
                reward=reward,
            )

            if done:
                break

        # ── 结尾 ──────────────────────────────────────────────────────────
        final_reward, final_done, final_feedback = env.feedback()
        self.notify_observers(final_feedback)
        self.meta_memory.add_experiential_memory(
            label=final_done,
            feedback=final_feedback,
        )

        # ── Token 快照 ────────────────────────────────────────────────────
        t = token_tracker.summary()
        self.notify_observers(
            f"Token usage — "
            f"solver: {t['solver']['total']}  "
            f"env: {t['env']['total']}  "
            f"memory: {t['memory']['total']}  "
            f"total: {t['total']['total']}"
        )

        return final_reward, final_done

    # ── Observer ──────────────────────────────────────────────────────────────

    def add_observer(self, observer) -> None:
        self.observers.append(observer)

    def notify_observers(self, message: str) -> None:
        for observer in self.observers:
            observer.log(message)