import numpy as np


class RewardEngine:
    def __init__(self, sparse_reward_max=1.0):
        self.step_penalty = -0.01
        self.key_bonus = 0.5
        self.progress_bonus = 0.05

        self.sparse_reward_max = sparse_reward_max
        self.llm_total_budget = self.sparse_reward_max * 0.49
        self.llm_step_min = -0.05
        self.llm_step_max = 0.5
        self.reset()

    def reset(self):
        """Reset episode-level reward tracking state."""
        self.llm_accumulated_reward = 0.0
        self.prev_distance = None
        self.closest_distance = float("inf")
        self.visited_positions = set()  # Track revisits to discourage loops.

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def compute_dense_reward(self, info, prev_info):
        """Compute hand-crafted dense reward components for the current transition."""
        reward = self.step_penalty
        agent_pos = info["agent_pos"]

        # Penalize revisiting the same location.
        if agent_pos in self.visited_positions:
            reward -= 0.02
        else:
            self.visited_positions.add(agent_pos)

        target_pos = info["exit_pos"] if info["has_key"] else info["key_pos"]
        current_distance = self._manhattan_distance(agent_pos, target_pos)

        if current_distance < self.closest_distance:
            reward += self.progress_bonus
            self.closest_distance = current_distance

        self.prev_distance = current_distance

        if info["has_key"] and not prev_info["has_key"]:
            reward += self.key_bonus
            self.closest_distance = self._manhattan_distance(agent_pos, info["exit_pos"])
            self.visited_positions.clear()

        return reward

    def apply_llm_bounds(self, raw_llm_reward):
        """Clamp and budget the cumulative LLM shaping reward."""
        bounded_reward = np.clip(raw_llm_reward, self.llm_step_min, self.llm_step_max)
        if self.llm_accumulated_reward + abs(bounded_reward) > self.llm_total_budget:
            remaining = max(0, self.llm_total_budget - self.llm_accumulated_reward)
            bounded_reward = np.sign(bounded_reward) * remaining
        self.llm_accumulated_reward += abs(bounded_reward)
        return float(bounded_reward)
