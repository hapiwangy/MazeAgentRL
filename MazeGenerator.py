import argparse
import json
import os
import random
import numpy as np
from collections import deque
<<<<<<< HEAD

class MazeGenerator:
    '''
    Maze cell representations:
    0: passage
    1: wall
    2: entrance
    3: exit
    4: key
    5: agent (used only in visualization, not in the grid data)
    '''
    def __init__(self):
        pass

    def generate_dfs_maze(self, size):
        """Standard Depth-First Search for long, winding paths."""
        maze = np.ones((size, size), dtype=int)
        stack = [(1, 1)]
        maze[1, 1] = 0

        while stack:
            r, c = stack[-1]
            directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
            random.shuffle(directions)
            
            found = False
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 < nr < size - 1 and 0 < nc < size - 1 and maze[nr, nc] == 1:
                    maze[r + dr // 2, c + dc // 2] = 0
                    maze[nr, nc] = 0
                    stack.append((nr, nc))
                    found = True
                    break
            if not found:
                stack.pop()
        return maze

    def generate_prim_maze(self, size):
        """Prim's Algorithm for more branching and short dead-ends."""
        maze = np.ones((size, size), dtype=int)
        start_r, start_c = 1, 1
        maze[start_r, start_c] = 0
        
        # Walls list: (wall_r, wall_c, passage_r, passage_c)
        # We store the wall to break and the cell it leads to.
        walls = []
        for dr, dc in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
            nr, nc = start_r + dr, start_c + dc
            if 0 < nr < size - 1 and 0 < nc < size - 1:
                walls.append((start_r + dr // 2, start_c + dc // 2, nr, nc))

        while walls:
            wr, wc, pr, pc = walls.pop(random.randint(0, len(walls) - 1))

            if maze[pr, pc] == 1:
                maze[wr, wc] = 0
                maze[pr, pc] = 0
                for dr, dc in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                    nr, nc = pr + dr, pc + dc
                    if 0 < nr < size - 1 and 0 < nc < size - 1 and maze[nr, nc] == 1:
                        walls.append((pr + dr // 2, pc + dc // 2, nr, nc))
        return maze

    def is_solvable(self, maze_grid, start, key, exit_pos):
        """Checks if a path exists: Start -> Key -> Exit."""
        def bfs(s, target_val):
            q = deque([s])
            visited = {s}
            while q:
                r, c = q.popleft()
                if maze_grid[r][c] == target_val:
                    return (r, c)
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < len(maze_grid) and 0 <= nc < len(maze_grid) and 
                        maze_grid[nr][nc] != 1 and (nr, nc) not in visited):
                        visited.add((nr, nc))
                        q.append((nr, nc))
            return None

        # Step 1: Can we reach the Key?
        key_found_pos = bfs(start, 4)
        if not key_found_pos: return False
        
        # Step 2: Can we reach the Exit from the Key?
        exit_found_pos = bfs(key_found_pos, 3)
        return exit_found_pos is not None

    def generate_single_maze(self, size, method="dfs"):
        """Ties generation, hole-punching, and validation together."""
        if size % 2 == 0: raise ValueError("Size must be odd.")
        
        while True:
            # 1. Generate Base
            if method == "prim":
                maze = self.generate_prim_maze(size)
            else:
                maze = self.generate_dfs_maze(size)

            # 2. Optimized Hole Punching
            # Find all internal walls (excluding the outer boundary)
            wall_indices = np.argwhere(maze[1:-1, 1:-1] == 1) + 1
            num_holes = min(len(wall_indices), size) 
            if num_holes > 0:
                indices = np.random.choice(len(wall_indices), num_holes, replace=False)
                for idx in indices:
                    r, c = wall_indices[idx]
                    maze[r, c] = 0

            # 3. Place Entities
            empty_cells = list(zip(*np.where(maze == 0)))
            if len(empty_cells) < 3: continue
            
            s, e, k = random.sample(empty_cells, 3)
            maze[s], maze[e], maze[k] = 2, 3, 4
            
            # 4. Validity Check
            grid_list = maze.tolist()
            if self.is_solvable(grid_list, s, k, e):
                return grid_list

=======

class MazeGenerator:
    '''
    Maze cell representations:
    0: passage
    1: wall
    2: entrance
    3: exit
    4: key
    5: agent (used only in visualization, not in the grid data)
    '''
    def __init__(self):
        pass

    def generate_dfs_maze(self, size):
        """Standard Depth-First Search for long, winding paths."""
        maze = np.ones((size, size), dtype=int)
        stack = [(1, 1)]
        maze[1, 1] = 0

        while stack:
            r, c = stack[-1]
            directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
            random.shuffle(directions)
            
            found = False
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 < nr < size - 1 and 0 < nc < size - 1 and maze[nr, nc] == 1:
                    maze[r + dr // 2, c + dc // 2] = 0
                    maze[nr, nc] = 0
                    stack.append((nr, nc))
                    found = True
                    break
            if not found:
                stack.pop()
        return maze

    def generate_prim_maze(self, size):
        """Prim's Algorithm for more branching and short dead-ends."""
        maze = np.ones((size, size), dtype=int)
        start_r, start_c = 1, 1
        maze[start_r, start_c] = 0
        
        # Walls list: (wall_r, wall_c, passage_r, passage_c)
        # We store the wall to break and the cell it leads to.
        walls = []
        for dr, dc in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
            nr, nc = start_r + dr, start_c + dc
            if 0 < nr < size - 1 and 0 < nc < size - 1:
                walls.append((start_r + dr // 2, start_c + dc // 2, nr, nc))

        while walls:
            wr, wc, pr, pc = walls.pop(random.randint(0, len(walls) - 1))

            if maze[pr, pc] == 1:
                maze[wr, wc] = 0
                maze[pr, pc] = 0
                for dr, dc in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                    nr, nc = pr + dr, pc + dc
                    if 0 < nr < size - 1 and 0 < nc < size - 1 and maze[nr, nc] == 1:
                        walls.append((pr + dr // 2, pc + dc // 2, nr, nc))
        return maze

    def is_solvable(self, maze_grid, start, key, exit_pos):
        """Checks if a path exists: Start -> Key -> Exit."""
        def bfs(s, target_val):
            q = deque([s])
            visited = {s}
            while q:
                r, c = q.popleft()
                if maze_grid[r][c] == target_val:
                    return (r, c)
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < len(maze_grid) and 0 <= nc < len(maze_grid) and 
                        maze_grid[nr][nc] != 1 and (nr, nc) not in visited):
                        visited.add((nr, nc))
                        q.append((nr, nc))
            return None

        # Step 1: Can we reach the Key?
        key_found_pos = bfs(start, 4)
        if not key_found_pos: return False
        
        # Step 2: Can we reach the Exit from the Key?
        exit_found_pos = bfs(key_found_pos, 3)
        return exit_found_pos is not None

    def generate_single_maze(self, size, method="dfs"):
        """Ties generation, hole-punching, and validation together."""
        if size % 2 == 0: raise ValueError("Size must be odd.")
        
        while True:
            # 1. Generate Base
            if method == "prim":
                maze = self.generate_prim_maze(size)
            else:
                maze = self.generate_dfs_maze(size)

            # 2. Optimized Hole Punching
            # Find all internal walls (excluding the outer boundary)
            wall_indices = np.argwhere(maze[1:-1, 1:-1] == 1) + 1
            num_holes = min(len(wall_indices), size) 
            if num_holes > 0:
                indices = np.random.choice(len(wall_indices), num_holes, replace=False)
                for idx in indices:
                    r, c = wall_indices[idx]
                    maze[r, c] = 0

            # 3. Place Entities
            empty_cells = list(zip(*np.where(maze == 0)))
            if len(empty_cells) < 3: continue
            
            s, e, k = random.sample(empty_cells, 3)
            maze[s], maze[e], maze[k] = 2, 3, 4
            
            # 4. Validity Check
            grid_list = maze.tolist()
            if self.is_solvable(grid_list, s, k, e):
                return grid_list

>>>>>>> 782edc09766074deaca156230cc233e6bcb4b88a
    def build_dataset(self, num_train=1000, num_val=200, num_test=200, sizes=None, output_dir="dataset", output_suffix=""):
        if sizes is None:
            sizes = [9, 25]
        if not sizes:
            raise ValueError("At least one maze size must be provided.")
        if any(size % 2 == 0 for size in sizes):
            raise ValueError("All maze sizes must be odd.")

        os.makedirs(output_dir, exist_ok=True)
        splits = {"train": num_train, "val": num_val, "test": num_test}

        for split_name, count in splits.items():
            print(f"Generating {split_name}...")
            dataset = []
            for i in range(count):
                size = sizes[i % len(sizes)]
                # Alternate between DFS and Prim
                algo = "dfs" if i % 2 == 0 else "prim"
                grid = self.generate_single_maze(size, method=algo)
                dataset.append({"id": f"{split_name}_{i}", "size": size, "algo": algo, "grid": grid})

            filename = f"{split_name}{output_suffix}.json"
            with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
                json.dump(dataset, f, separators=(",", ":"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate solvable maze datasets.")
    parser.add_argument("--train", type=int, default=1000, help="Number of training mazes.")
    parser.add_argument("--val", type=int, default=200, help="Number of validation mazes.")
    parser.add_argument("--test", type=int, default=200, help="Number of test mazes.")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[9, 25],
        help="Odd maze sizes to generate, e.g. --sizes 15 17",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset",
        help="Directory where dataset json files are written.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Optional suffix appended to output filenames, e.g. _15_17.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Optional sample maze size to print. Defaults to the first provided size.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    generator = MazeGenerator()

    # Set seeds for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)

    sample_size = args.sample_size or args.sizes[0]
    sample = generator.generate_single_maze(sample_size)
    print(f"=== Sample {sample_size}x{sample_size} Maze ===")
    chars = {0: " ", 1: "#", 2: "S", 3: "E", 4: "K"}
    for row in sample:
        print("".join([chars[val] for val in row]))
    print("\n")

    # Build the full dataset used by the project.
    generator.build_dataset(
        num_train=args.train,
        num_val=args.val,
        num_test=args.test,
        sizes=args.sizes,
        output_dir=args.output_dir,
        output_suffix=args.output_suffix,
    )
