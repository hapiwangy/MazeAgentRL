import random
import numpy as np
from collections import deque

from reward_config import DENSE_REWARD_CONFIG, LLM_REWARD_RANGE_CONFIG, SPARSE_REWARD_VALUES

class RewardEngine:
    def __init__(self):
        # Sparse Rewards (Used within Dense calculations)
        self.sparse_key_reward = SPARSE_REWARD_VALUES["key"]
        self.sparse_exit_reward = SPARSE_REWARD_VALUES["exit"]

        # Dense Config (General)
        self.step_penalty = DENSE_REWARD_CONFIG["step_penalty"]
        self.revisit_penalty = DENSE_REWARD_CONFIG["revisit_penalty"]
        self.progress_scale = DENSE_REWARD_CONFIG["progress_scale"]

        # Weighting for the Weighted Distance logic
        self.pre_key_key_weight = DENSE_REWARD_CONFIG["pre_key_key_weight"]
        self.pre_key_exit_weight = DENSE_REWARD_CONFIG["pre_key_exit_weight"]
        self.post_key_key_weight = DENSE_REWARD_CONFIG["post_key_key_weight"]
        self.post_key_exit_weight = DENSE_REWARD_CONFIG["post_key_exit_weight"]

        # LLM Budgeting
        self.sparse_reward_max = self.sparse_exit_reward
        self.llm_total_budget = self.sparse_reward_max * LLM_REWARD_RANGE_CONFIG["total_budget_ratio"]
        self.llm_step_min = LLM_REWARD_RANGE_CONFIG["step_min"]
        self.llm_step_max = LLM_REWARD_RANGE_CONFIG["step_max"]

        # Distance Maps (Pre-calculated per maze)
        self.key_dist_map = None
        self.exit_dist_map = None
        
        self.reset()

    def reset(self):
        """Reset episode-level reward tracking state."""
        self.llm_accumulated_reward = 0.0
        self.visited_positions = set()

    def initialize_episode(self, maze_grid, key_pos, exit_pos):
        """Pre-calculate path distances to targets using BFS."""
        self.reset()
        maze_np = np.array(maze_grid)
        self.key_dist_map = self._compute_bfs_dist_map(maze_np, tuple(key_pos))
        self.exit_dist_map = self._compute_bfs_dist_map(maze_np, tuple(exit_pos))

    def _compute_bfs_dist_map(self, maze, target):
        """Shortest path distance map (BFS). Walls (1) are untraversable."""
        dist_map = np.full(maze.shape, 999, dtype=int)
        dist_map[target] = 0
        queue = deque([target])

        while queue:
            r, c = queue.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < maze.shape[0] and 0 <= nc < maze.shape[1]:
                    if maze[nr, nc] != 1 and dist_map[nr, nc] == 999:
                        dist_map[nr, nc] = dist_map[r, c] + 1
                        queue.append((nr, nc))
        return dist_map

    def _weighted_path_distance(self, info):
        """Calculates a distance metric based on BFS paths and state-specific weights."""
        r, c = info["agent_pos"]
        key_dist = self.key_dist_map[r, c]
        exit_dist = self.exit_dist_map[r, c]

        if info["has_key"]:
            w_k, w_e = self.post_key_key_weight, self.post_key_exit_weight
        else:
            w_k, w_e = self.pre_key_key_weight, self.pre_key_exit_weight

        return (w_k * key_dist) + (w_e * exit_dist)

    def compute_sparse_reward(self, info, prev_info):
        """Return milestone sparse rewards for key pickup and successful exit."""
        reward = 0.0
        if info["has_key"] and not prev_info["has_key"]:
            reward += self.sparse_key_reward
        if info.get("is_success", False):
            reward += self.sparse_exit_reward
        return reward

    def compute_dense_reward(self, info, prev_info):
        """
        The 'All-in-One' Dense Reward.
        Awards milestones (Key/Exit) AND shaping (Path Progress/Penalties).
        """
        reward = self.step_penalty
        agent_pos = tuple(info["agent_pos"])

        # 1. Milestones (Obtaining Key and Exiting)
        if info["has_key"] and not prev_info["has_key"]:
            reward += self.sparse_key_reward
            self.visited_positions.clear() # Allow revisiting to find the exit

        if info.get("is_success", False):
            reward += self.sparse_exit_reward

        # 2. Revisit Penalty
        if agent_pos in self.visited_positions:
            reward += self.revisit_penalty
        else:
            self.visited_positions.add(agent_pos)

        # 3. Path-Aware Shaping (Potential-Based Progress)
        # We only apply shaping if the goal state (key status) remains the same.
        # This prevents the weight-shift from causing a 'fake' reward spike.
        if info["has_key"] == prev_info["has_key"]:
            dist_prev = self._weighted_path_distance(prev_info)
            dist_curr = self._weighted_path_distance(info)
            progress = dist_prev - dist_curr
            reward += self.progress_scale * progress

        return reward

    def sample_llm_reward(self, raw_llm_range, *, scale=1.0, budget_scale=1.0, deterministic=False):
        """Clamp the LLM range and sample while enforcing the episode budget.

        Args:
            raw_llm_range: Dict with "min" and "max" keys.
            scale: Multiplier applied to the sampled reward (used for llm-only mode).
            budget_scale: Multiplier applied to the per-episode absolute reward budget.
        """
        low = np.clip(raw_llm_range["min"], self.llm_step_min, self.llm_step_max)
        high = np.clip(raw_llm_range["max"], self.llm_step_min, self.llm_step_max)

        if low > high: low, high = high, low
        if deterministic:
            candidate = 0.5 * (low + high)
        else:
            candidate = random.uniform(low, high)
        candidate = float(candidate) * float(scale)

        effective_budget = self.llm_total_budget * float(budget_scale)
        if self.llm_accumulated_reward + abs(candidate) > effective_budget:
            remaining = max(0.0, effective_budget - self.llm_accumulated_reward)
            candidate = float(np.sign(candidate) * remaining)

        self.llm_accumulated_reward += abs(candidate)
        return float(candidate)