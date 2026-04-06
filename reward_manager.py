from OpenAILLM import OpenAILLM
from RewardEngine import RewardEngine
from reward_config import build_reward_components, combine_rewards, reward_mode_uses_llm


class RewardManager:
    """Centralized reward pipeline for sparse, dense, and LLM-based shaping."""

    def __init__(self, reward_mode, llm_model_name="gpt-4o-mini"):
        self.reward_mode = reward_mode
        self.reward_engine = RewardEngine()
        self.llm_api = OpenAILLM(model_name=llm_model_name) if reward_mode_uses_llm(reward_mode) else None

    def reset(self):
        self.reward_engine.reset()

    def compute_step_reward(self, current_info, prev_info):
        sparse_reward = self.reward_engine.compute_sparse_reward(current_info, prev_info)
        dense_reward = self.reward_engine.compute_dense_reward(current_info, prev_info)

        llm_reward_range = {"min": 0.0, "max": 0.0, "thought_process": "LLM disabled"}
        llm_reward = 0.0
        if self.llm_api is not None:
            llm_reward_range = self.llm_api.get_reward_range(current_info, prev_info)
            llm_reward = self.reward_engine.sample_llm_reward(llm_reward_range)

        reward_components = build_reward_components(
            sparse_reward=sparse_reward,
            dense_reward=dense_reward,
            llm_reward=llm_reward,
        )
        total_reward = combine_rewards(self.reward_mode, reward_components)

        return total_reward, reward_components, llm_reward_range
