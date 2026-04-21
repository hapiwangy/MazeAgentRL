from QwenLLM import QwenLLM
from RewardEngine import RewardEngine
from reward_config import (
    LLM_REWARD_RANGE_CONFIG,
    build_reward_components,
    combine_rewards,
    reward_mode_uses_llm,
)


class RewardManager:
    """Centralized reward pipeline for sparse, dense, and LLM-based shaping."""

    def __init__(self, reward_mode, llm_model_name="Qwen/Qwen2.5-7B-Instruct"):
        self.reward_mode = reward_mode
        self.reward_engine = RewardEngine()
        self.llm_api = QwenLLM(model_name=llm_model_name) if reward_mode_uses_llm(reward_mode) else None

    def reset(self):
        self.reward_engine.reset()

    def compute_step_reward(self, current_info, prev_info):
        sparse_reward = self.reward_engine.compute_sparse_reward(current_info, prev_info)
        dense_reward = self.reward_engine.compute_dense_reward(current_info, prev_info)

        llm_reward_range = {"min": 0.0, "max": 0.0, "state_analysis": "LLM disabled"}
        llm_reward = 0.0
        if self.llm_api is not None:
            llm_reward_range = self.llm_api.get_reward_range(current_info, prev_info)
            llm_only = self.reward_mode == "llm"
            llm_reward = self.reward_engine.sample_llm_reward(
                llm_reward_range,
                scale=LLM_REWARD_RANGE_CONFIG["llm_only_scale"] if llm_only else 1.0,
                budget_scale=LLM_REWARD_RANGE_CONFIG["llm_only_budget_scale"] if llm_only else 1.0,
                deterministic=llm_only,
            )

            # In llm-only mode, the agent otherwise never sees large milestone signals.
            # Add small deterministic bonuses to make the objective learnable.
            if llm_only:
                if current_info.get("has_key", False) and not prev_info.get("has_key", False):
                    llm_reward += float(LLM_REWARD_RANGE_CONFIG["llm_only_key_bonus"])
                if current_info.get("is_success", False):
                    llm_reward += float(LLM_REWARD_RANGE_CONFIG["llm_only_exit_bonus"])

                agent_pos = tuple(current_info.get("agent_pos", ()))
                exit_pos = tuple(current_info.get("exit_pos", ()))
                if agent_pos and exit_pos and (agent_pos == exit_pos) and not current_info.get("has_key", False):
                    llm_reward += float(LLM_REWARD_RANGE_CONFIG["llm_only_exit_without_key_penalty"])

        reward_components = build_reward_components(
            sparse_reward=sparse_reward,
            dense_reward=dense_reward,
            llm_reward=llm_reward,
        )
        total_reward = combine_rewards(self.reward_mode, reward_components)

        return total_reward, reward_components, llm_reward_range
