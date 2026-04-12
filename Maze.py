import numpy as np
import gymnasium as gym
from gymnasium import spaces
from reward_config import SPARSE_REWARD_VALUES

class MazeEnv(gym.Env):
    def __init__(self, maze_map, max_steps=100):
        super(MazeEnv, self).__init__()
        self.initial_map = np.array(maze_map)
        self.grid_size = self.initial_map.shape
        self.max_steps = max_steps
        
        # 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)  
        
        # 3x3 local grid of integers (0-5)
        self.observation_space = spaces.Box(low=0, high=5, shape=(3, 3), dtype=np.int8)

        # Locate key elements once at initialization
        self.start_pos = tuple(np.argwhere(self.initial_map == 2)[0])
        self.exit_pos = tuple(np.argwhere(self.initial_map == 3)[0])
        self.key_pos = tuple(np.argwhere(self.initial_map == 4)[0])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start_pos
        self.has_key = False
        self.current_step = 0
        self.current_map = self.initial_map.copy()
        
        # Return observation and info
        return self._get_local_obs(), self._get_info()

    def step(self, action):
        self.current_step += 1
        row, col = self.agent_pos

        # Action mapping
        if action == 0: row -= 1    # up
        elif action == 1: row += 1  # down
        elif action == 2: col -= 1  # left
        elif action == 3: col += 1  # right

        # Collision detection (Walls = 1)
        if (0 <= row < self.grid_size[0] and 
            0 <= col < self.grid_size[1] and 
            self.current_map[row, col] != 1):
            self.agent_pos = (row, col)

        picked_key = False
        # Pickup key logic
        if self.agent_pos == self.key_pos and not self.has_key:
            self.has_key = True
            picked_key = True
            # Transform key tile into path once picked up
            self.current_map[self.key_pos] = 0 

        # Success condition: at exit AND has key
        is_success = (self.agent_pos == self.exit_pos) and self.has_key
        
        terminated = is_success
        truncated = self.current_step >= self.max_steps
        
        # The RewardEngine usually handles rewards now, but we keep 
        # the base sparse rewards here for env consistency.
        reward = 0.0
        if picked_key:
            reward += SPARSE_REWARD_VALUES["key"]
        if is_success:
            reward += SPARSE_REWARD_VALUES["exit"]

        info = self._get_info()
        info["is_success"] = is_success
        # Specifically adding this for the RewardManager/Agent
        info["picked_key_this_step"] = picked_key 

        return self._get_local_obs(), reward, terminated, truncated, info

    def _get_local_obs(self):
        """Returns a 3x3 window around the agent."""
        row, col = self.agent_pos
        # Default to walls (1) for out-of-bounds padding
        obs = np.ones((3, 3), dtype=np.int8) 
        
        for i in range(3):
            for j in range(3):
                r, c = row - 1 + i, col - 1 + j
                if 0 <= r < self.grid_size[0] and 0 <= c < self.grid_size[1]:
                    obs[i, j] = self.current_map[r, c]
        
        # Agent marker (5) always at center
        obs[1, 1] = 5 
        return obs

    def _get_info(self):
        """Metadata for RewardManager and tracking."""
        return {
            "agent_pos": self.agent_pos,
            "has_key": self.has_key,
            "exit_pos": self.exit_pos,
            "key_pos": self.key_pos,
            "step_count": self.current_step,
            "global_map_string": self._get_global_state_string(),
        }

    def _get_global_state_string(self):
        chars = {0: ".", 1: "#", 2: "S", 3: "E", 4: "K", 5: "A"}
        disp = self.current_map.copy()
        
        # Overlay fixed markers
        disp[self.start_pos] = 2
        disp[self.exit_pos] = 3
        if not self.has_key:
            disp[self.key_pos] = 4
        
        disp[self.agent_pos] = 5

        res = ""
        for row in disp:
            res += " ".join([chars[val] for val in row]) + "\n"
        return res.strip()