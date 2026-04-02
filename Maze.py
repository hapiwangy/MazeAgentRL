import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MazeEnv(gym.Env):
    def __init__(self, maze_map, max_steps=100):
        super(MazeEnv, self).__init__()
        self.initial_map = np.array(maze_map)
        self.grid_size = self.initial_map.shape
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(4)  # 0: up, 1: down, 2: left, 3: right
        self.observation_space = spaces.Box(low=0, high=5, shape=(3, 3), dtype=np.int8)

        self.start_pos = tuple(np.argwhere(self.initial_map == 2)[0])
        self.exit_pos = tuple(np.argwhere(self.initial_map == 3)[0])
        self.key_pos = tuple(np.argwhere(self.initial_map == 4)[0])

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

        # Action mapping.
        if action == 0:
            row -= 1
        elif action == 1:
            row += 1
        elif action == 2:
            col -= 1
        elif action == 3:
            col += 1

        # Prevent movement into walls or outside the grid.
        if (
            0 <= row < self.grid_size[0]
            and 0 <= col < self.grid_size[1]
            and self.current_map[row, col] != 1
        ):
            self.agent_pos = (row, col)

        # Pick up the key when the agent reaches it.
        if self.agent_pos == self.key_pos and not self.has_key:
            self.has_key = True
            self.current_map[self.key_pos] = 0

        is_success = (self.agent_pos == self.exit_pos) and self.has_key
        terminated = is_success
        truncated = self.current_step >= self.max_steps
        reward = 1.0 if is_success else 0.0  # Sparse reward

        info = self._get_info()
        info["is_success"] = is_success
        return self._get_local_obs(), reward, terminated, truncated, info

    def _get_local_obs(self):
        row, col = self.agent_pos
        obs = np.ones((3, 3), dtype=np.int8)
        for i in range(3):
            for j in range(3):
                r, c = row - 1 + i, col - 1 + j
                if 0 <= r < self.grid_size[0] and 0 <= c < self.grid_size[1]:
                    obs[i, j] = self.current_map[r, c]
        obs[1, 1] = 5  # The agent is always centered in the local view.
        return obs

    def _get_info(self):
        return {
            "agent_pos": self.agent_pos,
            "has_key": self.has_key,
            "exit_pos": self.exit_pos,
            "key_pos": self.key_pos,
            "global_map_string": self._get_global_state_string(),
        }

    def _get_global_state_string(self):
        """Convert the map to a string: A=agent, K=key, E=exit, S=start, #=wall, .=path."""
        chars = {0: ".", 1: "#", 2: "S", 3: "E", 4: "K", 5: "A"}
        disp = self.current_map.copy()

        # Ensure the fixed tiles remain visible.
        disp[self.start_pos] = 2
        disp[self.exit_pos] = 3
        if not self.has_key:
            disp[self.key_pos] = 4

        # The agent marker overrides the underlying tile.
        disp[self.agent_pos] = 5

        res = ""
        for row in disp:
            res += " ".join([chars[val] for val in row]) + "\n"
        return res.strip()


if __name__ == "__main__":
    # Define a 9x9 test maze.
    # 0: path, 1: wall, 2: start, 3: exit, 4: key
    map_9x9 = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 0, 0, 1, 0, 0, 3, 1],
        [1, 1, 1, 0, 1, 0, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 4, 1, 0, 0, 1],
        [1, 1, 1, 1, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]

    env = MazeEnv(maze_map=map_9x9)
    obs, info = env.reset()

    print("=== Initial State ===")
    print(info["global_map_string"])
    print("\nInitial 3x3 local view (5 is the agent, 1 is a wall):")
    print(obs)

    # Let the agent move randomly for 3 steps.
    action_names = {0: "up", 1: "down", 2: "left", 3: "right"}

    for step in range(1, 4):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\n=== Step {step}: attempt to move {action_names[action]} ===")
        print(f"Current position: {info['agent_pos']} | Has key: {info['has_key']}")
        print("Global map:")
        print(info["global_map_string"])
        print("Local view:")
        print(obs)
