import hashlib

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from reward_config import SPARSE_REWARD_VALUES


class MazeEnv(gym.Env):
    def __init__(self, maze_map, max_steps=100, include_text_info=False):
        super().__init__()
        self.initial_map = np.asarray(maze_map, dtype=np.int8)
        self.grid_size = self.initial_map.shape
        self.max_steps = int(max_steps)
        self.include_text_info = bool(include_text_info)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=5, shape=(3, 3), dtype=np.int8)

        self.start_pos = tuple(np.argwhere(self.initial_map == 2)[0])
        self.exit_pos = tuple(np.argwhere(self.initial_map == 3)[0])
        self.key_pos = tuple(np.argwhere(self.initial_map == 4)[0])

        self.maze_layout_string = self._build_maze_layout_string()
        self.maze_layout_hash = hashlib.sha1(self.maze_layout_string.encode("utf-8")).hexdigest()[:10]
        self.traversable_map = np.where(self.initial_map == 1, 1, 0).astype(np.int8)
        self.current_map = self.initial_map.copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start_pos
        self.has_key = False
        self.current_step = 0
        self.current_map = self.initial_map.copy()
        return self._get_local_obs(), self._get_info()

    def step(self, action):
        self.current_step += 1
        row, col = self.agent_pos

        if action == 0:
            row -= 1
        elif action == 1:
            row += 1
        elif action == 2:
            col -= 1
        elif action == 3:
            col += 1

        if 0 <= row < self.grid_size[0] and 0 <= col < self.grid_size[1] and self.current_map[row, col] != 1:
            self.agent_pos = (row, col)

        picked_key = False
        if self.agent_pos == self.key_pos and not self.has_key:
            self.has_key = True
            picked_key = True
            self.current_map[self.key_pos] = 0

        is_success = (self.agent_pos == self.exit_pos) and self.has_key
        terminated = is_success
        truncated = self.current_step >= self.max_steps

        reward = 0.0
        if picked_key:
            reward += SPARSE_REWARD_VALUES["key"]
        if is_success:
            reward += SPARSE_REWARD_VALUES["exit"]

        info = self._get_info()
        info["is_success"] = is_success
        info["picked_key_this_step"] = picked_key

        return self._get_local_obs(), reward, terminated, truncated, info

    def _get_local_obs(self):
        row, col = self.agent_pos
        obs = np.ones((3, 3), dtype=np.int8)

        row_start = max(0, row - 1)
        row_end = min(self.grid_size[0], row + 2)
        col_start = max(0, col - 1)
        col_end = min(self.grid_size[1], col + 2)

        obs_row_start = 1 - (row - row_start)
        obs_col_start = 1 - (col - col_start)
        obs_row_end = obs_row_start + (row_end - row_start)
        obs_col_end = obs_col_start + (col_end - col_start)

        obs[obs_row_start:obs_row_end, obs_col_start:obs_col_end] = self.current_map[row_start:row_end, col_start:col_end]
        obs[1, 1] = 5
        return obs

    def _get_info(self):
        info = {
            "agent_pos": self.agent_pos,
            "has_key": self.has_key,
            "exit_pos": self.exit_pos,
            "key_pos": self.key_pos,
            "step_count": self.current_step,
            "maze_grid": self.initial_map,
            "maze_layout_hash": self.maze_layout_hash,
            "local_obs": self._get_local_obs(),
        }

        if self.include_text_info:
            info["maze_layout_string"] = self.maze_layout_string
            info["global_map_string"] = self._get_global_state_string()

        return info

    def _build_maze_layout_string(self):
        chars = {0: ".", 1: "#", 2: "S", 3: "E", 4: "K"}
        return "\n".join(" ".join(chars[int(val)] for val in row) for row in self.initial_map)

    def _get_global_state_string(self):
        chars = {0: ".", 1: "#", 2: "S", 3: "E", 4: "K", 5: "A"}
        disp = self.current_map.copy()
        disp[self.start_pos] = 2
        disp[self.exit_pos] = 3
        if not self.has_key:
            disp[self.key_pos] = 4
        disp[self.agent_pos] = 5
        return "\n".join(" ".join(chars[int(val)] for val in row) for row in disp)
