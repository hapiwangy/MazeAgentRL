import json
import os
import random

import numpy as np


class MazeGenerator:
    def __init__(self):
        pass

    def generate_single_maze(self, size):
        """
        Generate a random maze using depth-first search.

        The maze uses the following tile encoding:
        0: path, 1: wall, 2: start, 3: exit, 4: key
        """
        if size % 2 == 0:
            raise ValueError("Maze size must be an odd number.")

        # Initialize the map as all walls.
        maze = np.ones((size, size), dtype=int)

        # Start carving passages from cell (1, 1).
        stack = [(1, 1)]
        maze[1, 1] = 0

        while stack:
            r, c = stack[-1]
            neighbors = []

            # Move two cells at a time to preserve wall structure.
            directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
            random.shuffle(directions)

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 < nr < size - 1 and 0 < nc < size - 1 and maze[nr, nc] == 1:
                    neighbors.append((nr, nc, dr, dc))

            if neighbors:
                nr, nc, dr, dc = neighbors[0]
                # Carve the wall between the current cell and the next cell.
                maze[r + dr // 2, c + dc // 2] = 0
                maze[nr, nc] = 0
                stack.append((nr, nc))
            else:
                # Backtrack when no unvisited neighbor remains.
                stack.pop()

        # Add a small number of extra openings to reduce excessive linearity.
        extra_holes = size
        for _ in range(extra_holes):
            r, c = random.randint(1, size - 2), random.randint(1, size - 2)
            if maze[r, c] == 1:
                maze[r, c] = 0

        # Randomly place the start, exit, and key on empty cells.
        empty_cells = list(zip(*np.where(maze == 0)))
        start_pos, exit_pos, key_pos = random.sample(empty_cells, 3)

        maze[start_pos] = 2
        maze[exit_pos] = 3
        maze[key_pos] = 4

        return maze.tolist()

    def build_dataset(self, num_train=1000, num_val=200, num_test=200):
        """
        Build train, validation, and test datasets with 9x9 and 25x25 mazes.
        """
        os.makedirs("dataset", exist_ok=True)

        splits = {
            "train": num_train,
            "val": num_val,
            "test": num_test,
        }

        for split_name, count in splits.items():
            print(f"Generating the {split_name} dataset ({count} mazes)...")
            dataset = []

            # Use 9x9 mazes for the first half and 25x25 mazes for the second half.
            for i in range(count):
                size = 9 if i < count / 2 else 25
                maze_grid = self.generate_single_maze(size)
                dataset.append(
                    {
                        "id": f"{split_name}_{size}x{size}_{i:04d}",
                        "size": size,
                        "grid": maze_grid,
                    }
                )

            filepath = os.path.join("dataset", f"{split_name}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(dataset, f, separators=(",", ":"))

            print(f"{split_name.capitalize()} dataset saved to {filepath}")


if __name__ == "__main__":
    generator = MazeGenerator()

    # Generate and display a sample 9x9 maze.
    sample = generator.generate_single_maze(9)
    print("=== Sample 9x9 Maze ===")
    chars = {0: " ", 1: "#", 2: "S", 3: "E", 4: "K"}
    for row in sample:
        print("".join([chars[val] for val in row]))
    print("\n")

    # Build the full dataset used by the project.
    generator.build_dataset(num_train=1000, num_val=200, num_test=200)
