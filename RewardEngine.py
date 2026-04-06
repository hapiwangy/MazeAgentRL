import random

import numpy as np

from reward_config import DENSE_REWARD_CONFIG, LLM_REWARD_RANGE_CONFIG, SPARSE_REWARD_VALUES


class RewardEngine:
    def __init__(self):
        self.sparse_key_reward = SPARSE_REWARD_VALUES["key"]
        self.sparse_exit_reward = SPARSE_REWARD_VALUES["exit"]

        self.step_penalty = DENSE_REWARD_CONFIG["step_penalty"]
        self.revisit_penalty = DENSE_REWARD_CONFIG["revisit_penalty"]
        self.progress_scale = DENSE_REWARD_CONFIG["progress_scale"]

        self.pre_key_key_weight = DENSE_REWARD_CONFIG["pre_key_key_weight"]
        self.pre_key_exit_weight = DENSE_REWARD_CONFIG["pre_key_exit_weight"]
        self.post_key_key_weight = DENSE_REWARD_CONFIG["post_key_key_weight"]
        self.post_key_exit_weight = DENSE_REWARD_CONFIG["post_key_exit_weight"]

        self.sparse_reward_max = self.sparse_exit_reward
        self.llm_total_budget = self.sparse_reward_max * LLM_REWARD_RANGE_CONFIG["total_budget_ratio"]
        self.llm_step_min = LLM_REWARD_RANGE_CONFIG["step_min"]
        self.llm_step_max = LLM_REWARD_RANGE_CONFIG["step_max"]
        self.reset()

    def reset(self):
        """Reset episode-level reward tracking state."""
        self.llm_accumulated_reward = 0.0
        self.visited_positions = set()  # Track revisits to discourage loops.

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def compute_sparse_reward(self, info, prev_info):
        """Return milestone sparse rewards for key pickup and successful exit."""
        reward = 0.0
        if info["has_key"] and not prev_info["has_key"]:
            reward += self.sparse_key_reward
        if info.get("is_success", False):
            reward += self.sparse_exit_reward
        return reward

    def _weighted_distance(self, info):
        agent_pos = info["agent_pos"]
        key_distance = self._manhattan_distance(agent_pos, info["key_pos"])
        exit_distance = self._manhattan_distance(agent_pos, info["exit_pos"])

        if info["has_key"]:
            key_weight = self.post_key_key_weight
            exit_weight = self.post_key_exit_weight
        else:
            key_weight = self.pre_key_key_weight
            exit_weight = self.pre_key_exit_weight

        return (key_weight * key_distance) + (exit_weight * exit_distance)

    def compute_dense_reward(self, info, prev_info):
        """Compute dense shaping from weighted distance progress to key and exit."""
        reward = self.step_penalty
        agent_pos = info["agent_pos"]

        # Penalize revisiting the same location.
        if agent_pos in self.visited_positions:
            reward += self.revisit_penalty
        else:
            self.visited_positions.add(agent_pos)

        previous_weighted_distance = self._weighted_distance(prev_info)
        current_weighted_distance = self._weighted_distance(info)
        progress = previous_weighted_distance - current_weighted_distance
        reward += self.progress_scale * progress

        if info["has_key"] and not prev_info["has_key"]:
            self.visited_positions.clear()

        return reward

    def sample_llm_reward(self, raw_llm_range):
        """Clamp the proposed LLM reward range and sample one reward value uniformly."""
        lower_bound = np.clip(raw_llm_range["min"], self.llm_step_min, self.llm_step_max)
        upper_bound = np.clip(raw_llm_range["max"], self.llm_step_min, self.llm_step_max)

        if lower_bound > upper_bound:
            lower_bound, upper_bound = upper_bound, lower_bound

        candidate_reward = random.uniform(lower_bound, upper_bound)
        if self.llm_accumulated_reward + abs(candidate_reward) > self.llm_total_budget:
            remaining = max(0, self.llm_total_budget - self.llm_accumulated_reward)
            candidate_reward = np.sign(candidate_reward) * remaining

        self.llm_accumulated_reward += abs(candidate_reward)
        return float(candidate_reward)
