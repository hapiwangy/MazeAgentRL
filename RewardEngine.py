import random
from collections import deque

import numpy as np

from reward_config import DENSE_REWARD_CONFIG, LLM_REWARD_RANGE_CONFIG, SPARSE_REWARD_VALUES


class RewardEngine:
    _distance_cache = {}

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

        self.key_dist_map = None
        self.exit_dist_map = None
        self.reset()

    def reset(self):
        self.llm_accumulated_reward = 0.0
        self.visited_positions = set()

    def initialize_episode(self, maze_grid, key_pos, exit_pos):
        self.reset()
        maze_np = np.asarray(maze_grid, dtype=np.int8)
        cache_key = (maze_np.shape, maze_np.tobytes(), tuple(key_pos), tuple(exit_pos))

        cached = self._distance_cache.get(cache_key)
        if cached is None:
            cached = (
                self._compute_bfs_dist_map(maze_np, tuple(key_pos)),
                self._compute_bfs_dist_map(maze_np, tuple(exit_pos)),
            )
            self._distance_cache[cache_key] = cached

        self.key_dist_map, self.exit_dist_map = cached

    def _compute_bfs_dist_map(self, maze, target):
        dist_map = np.full(maze.shape, 999, dtype=np.int16)
        dist_map[target] = 0
        queue = deque([target])

        while queue:
            r, c = queue.popleft()
            next_dist = dist_map[r, c] + 1
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < maze.shape[0] and 0 <= nc < maze.shape[1]:
                    if maze[nr, nc] != 1 and dist_map[nr, nc] == 999:
                        dist_map[nr, nc] = next_dist
                        queue.append((nr, nc))
        return dist_map

    def attach_distance_features(self, info):
        r, c = info["agent_pos"]
        info["key_distance"] = int(self.key_dist_map[r, c])
        info["exit_distance"] = int(self.exit_dist_map[r, c])
        return info

    def _weighted_path_distance(self, info):
        key_dist = info["key_distance"]
        exit_dist = info["exit_distance"]

        if info["has_key"]:
            w_k, w_e = self.post_key_key_weight, self.post_key_exit_weight
        else:
            w_k, w_e = self.pre_key_key_weight, self.pre_key_exit_weight

        return (w_k * key_dist) + (w_e * exit_dist)

    def compute_sparse_reward(self, info, prev_info):
        reward = 0.0
        if info["has_key"] and not prev_info["has_key"]:
            reward += self.sparse_key_reward
        if info.get("is_success", False):
            reward += self.sparse_exit_reward
        return reward

    def compute_dense_reward(self, info, prev_info):
        reward = self.step_penalty
        agent_pos = tuple(info["agent_pos"])

        if info["has_key"] and not prev_info["has_key"]:
            reward += self.sparse_key_reward
            self.visited_positions.clear()

        if info.get("is_success", False):
            reward += self.sparse_exit_reward

        if agent_pos in self.visited_positions:
            reward += self.revisit_penalty
        else:
            self.visited_positions.add(agent_pos)

        if info["has_key"] == prev_info["has_key"]:
            dist_prev = self._weighted_path_distance(prev_info)
            dist_curr = self._weighted_path_distance(info)
            reward += self.progress_scale * (dist_prev - dist_curr)

        return reward

    def sample_llm_reward(self, raw_llm_range, *, scale=1.0, budget_scale=1.0, deterministic=False):
        low = np.clip(raw_llm_range["min"], self.llm_step_min, self.llm_step_max)
        high = np.clip(raw_llm_range["max"], self.llm_step_min, self.llm_step_max)

        if low > high:
            low, high = high, low

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
