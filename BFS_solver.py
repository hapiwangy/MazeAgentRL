from collections import deque

import numpy as np


class BFSSolver:
    def __init__(self, maze_map):
        self.maze = np.array(maze_map)
        self.grid_size = self.maze.shape

        # Locate the key structural positions in the maze.
        self.start_pos = tuple(np.argwhere(self.maze == 2)[0])
        self.exit_pos = tuple(np.argwhere(self.maze == 3)[0])
        self.key_pos = tuple(np.argwhere(self.maze == 4)[0])

    def _bfs(self, start, target):
        """
        Run breadth-first search and return the shortest path length from start to target.
        """
        queue = deque([(start, 0)])
        visited = {start}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            curr_pos, steps = queue.popleft()

            if curr_pos == target:
                return steps

            for dr, dc in directions:
                r, c = curr_pos[0] + dr, curr_pos[1] + dc

                # Expand only valid, non-wall, and unvisited cells.
                if 0 <= r < self.grid_size[0] and 0 <= c < self.grid_size[1]:
                    if self.maze[r, c] != 1 and (r, c) not in visited:
                        visited.add((r, c))
                        queue.append(((r, c), steps + 1))

        # Return -1 when the target cannot be reached.
        return -1

    def get_optimal_steps(self):
        """
        Compute the shortest valid route: start -> key -> exit.
        """
        steps_to_key = self._bfs(self.start_pos, self.key_pos)
        if steps_to_key == -1:
            print("No valid path exists from the start to the key.")
            return -1

        steps_to_exit = self._bfs(self.key_pos, self.exit_pos)
        if steps_to_exit == -1:
            print("No valid path exists from the key to the exit.")
            return -1

        total_optimal_steps = steps_to_key + steps_to_exit

        return {
            "steps_to_key": steps_to_key,
            "steps_to_exit": steps_to_exit,
            "total_optimal_steps": total_optimal_steps,
        }


if __name__ == "__main__":
    # Example maze used to verify the BFS baseline.
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

    solver = BFSSolver(map_9x9)
    result = solver.get_optimal_steps()

    print("=== BFS Baseline Result ===")
    print(f"Shortest path from start to key: {result['steps_to_key']} steps")
    print(f"Shortest path from key to exit: {result['steps_to_exit']} steps")
    print(f"Total optimal path length: {result['total_optimal_steps']} steps")
