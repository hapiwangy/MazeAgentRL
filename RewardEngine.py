import random
import numpy as np
from collections import deque

from reward_config import DENSE_REWARD_CONFIG, LLM_REWARD_RANGE_CONFIG, SPARSE_REWARD_VALUES

class RewardEngine:
    def __init__(self):
        # Sparse Rewards
        self.sparse_key_reward = SPARSE_REWARD_VALUES["key"]
        self.sparse_exit_reward = SPARSE_REWARD_VALUES["exit"]

        # Dense Config (General)
        self.step_penalty = DENSE_REWARD_CONFIG["step_penalty"]
        self.revisit_penalty = DENSE_REWARD_CONFIG["revisit_penalty"]
        self.progress_scale = DENSE_REWARD_CONFIG["progress_scale"]

        # Dense Config (Weights)
        self.pre_key_key_weight = DENSE_REWARD_CONFIG["pre_key_key_weight"]
        self.pre_key_exit_weight = DENSE_REWARD_CONFIG["pre_key_exit_weight"]
        self.post_key_key_weight = DENSE_REWARD_CONFIG["post_key_key_weight"]
        self.post_key_exit_weight = DENSE_REWARD_CONFIG["post_key_exit_weight"]

        # LLM Budgeting
        self.sparse_reward_max = self.sparse_exit_reward
        self.llm_total_budget = self.sparse_reward_max * LLM_REWARD_RANGE_CONFIG["total_budget_ratio"]
        self.llm_step_min = LLM_REWARD_RANGE_CONFIG["step_min"]
        self.llm_step_max = LLM_REWARD_RANGE_CONFIG["step_max"]

        # Distance Maps (Set per episode)
        self.key_dist_map = None
        self.exit_dist_map = None
        
        self.reset()

    def reset(self):
        """Reset episode-level reward tracking state."""
        self.llm_accumulated_reward = 0.0
        self.visited_positions = set()

    def initialize_episode(self, maze_grid, key_pos, exit_pos):
        """
        Calculates BFS distance maps once at the start of an episode.
        Call this whenever a new maze is loaded.
        """
        self.reset()
        maze_np = np.array(maze_grid)
        self.key_dist_map = self._compute_bfs_dist_map(maze_np, tuple(key_pos))
        self.exit_dist_map = self._compute_bfs_dist_map(maze_np, tuple(exit_pos))

    def _compute_bfs_dist_map(self, maze, target):
        """Generates a grid of path-aware distances to the target."""
        dist_map = np.full(maze.shape, 999, dtype=int)
        dist_map[target] = 0
        queue = deque([target])

        while queue:
            r, c = queue.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < maze.shape[0] and 0 <= nc < maze.shape[1]:
                    # Cell must be path (0) or start/key/exit (2,4,3) - not wall (1)
                    if maze[nr, nc] != 1 and dist_map[nr, nc] == 999:
                        dist_map[nr, nc] = dist_map[r, c] + 1
                        queue.append((nr, nc))
        return dist_map

    def compute_sparse_reward(self, info, prev_info):
        """Return milestone sparse rewards for key pickup and successful exit."""
        reward = 0.0
        if info["has_key"] and not prev_info["has_key"]:
            reward += self.sparse_key_reward
        if info.get("is_success", False):
            reward += self.sparse_exit_reward
        return reward

    def _weighted_path_distance(self, info):
        """
        Calculates a weighted distance using actual path distances (BFS) 
        instead of Manhattan distances.
        """
        r, c = info["agent_pos"]
        # Use pre-calculated BFS distances
        key_distance = self.key_dist_map[r, c]
        exit_distance = self.exit_dist_map[r, c]

        if info["has_key"]:
            key_weight = self.post_key_key_weight
            exit_weight = self.post_key_exit_weight
        else:
            key_weight = self.pre_key_key_weight
            exit_weight = self.pre_key_exit_weight

        return (key_weight * key_distance) + (exit_weight * exit_distance)

    def compute_dense_reward(self, info, prev_info):
        """Compute dense shaping from weighted path distance progress."""
        reward = self.step_penalty
        agent_pos = tuple(info["agent_pos"])

        # 1. Penalize revisiting
        if agent_pos in self.visited_positions:
            reward += self.revisit_penalty
        else:
            self.visited_positions.add(agent_pos)

        # 2. Potential-Based Progress (using path-distance)
        # We wrap this in a potential check to ensure no cycles are rewarded
        previous_weighted_distance = self._weighted_path_distance(prev_info)
        current_weighted_distance = self._weighted_path_distance(info)
        
        # In PBRS: Reward = Phi(s') - Phi(s). 
        # Since smaller distance is higher potential, we use (prev - curr)
        progress = previous_weighted_distance - current_weighted_distance
        
        # Only reward progress if the 'has_key' state didn't change this step
        # to prevent a massive reward spike from switching weight profiles.
        if info["has_key"] == prev_info["has_key"]:
            reward += self.progress_scale * progress

        # 3. State transition management
        if info["has_key"] and not prev_info["has_key"]:
            self.visited_positions.clear()

        return reward

    def sample_llm_reward(self, raw_llm_range):
        """Clamp the proposed LLM reward range and sample within budget."""
        lower_bound = np.clip(raw_llm_range["min"], self.llm_step_min, self.llm_step_max)
        upper_bound = np.clip(raw_llm_range["max"], self.llm_step_min, self.llm_step_max)

        if lower_bound > upper_bound:
            lower_bound, upper_bound = upper_bound, lower_bound

        candidate_reward = random.uniform(lower_bound, upper_bound)
        
        # Maintain total budget for the episode
        if self.llm_accumulated_reward + abs(candidate_reward) > self.llm_total_budget:
            remaining = max(0, self.llm_total_budget - self.llm_accumulated_reward)
            candidate_reward = np.sign(candidate_reward) * remaining

        self.llm_accumulated_reward += abs(candidate_reward)
        return float(candidate_reward)