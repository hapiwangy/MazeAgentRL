# import json
# import os
# import time

# from openai import OpenAI

# from reward_config import LLM_REWARD_RANGE_CONFIG


# class OpenAILLM:
#     def __init__(self, model_name="gpt-4o-mini", cache_file="llm_cache.json"):
#         api_key = os.getenv("OPENAI_API_KEY")
#         if not api_key:
#             raise ValueError("OPENAI_API_KEY is not set. Please export it before running training.")

#         # Initialize the OpenAI client.
#         self.client = OpenAI(api_key=api_key)
#         self.model_name = model_name
#         self.min_reward = LLM_REWARD_RANGE_CONFIG["step_min"]
#         self.max_reward = LLM_REWARD_RANGE_CONFIG["step_max"]

#         # Cache LLM outputs to avoid repeated API calls for identical transitions.
#         self.cache_file = cache_file
#         self.cache = self._load_cache()
#         self.api_call_count = 0
#         self.cache_hit_count = 0

#     def _load_cache(self):
#         """Load the local reward cache if it is available."""
#         if os.path.exists(self.cache_file):
#             try:
#                 with open(self.cache_file, "r", encoding="utf-8") as f:
#                     return json.load(f)
#             except Exception as e:
#                 print(f"[Cache Load Error] {e}")
#         return {}

#     def _save_cache(self):
#         """Persist the reward cache to disk."""
#         try:
#             with open(self.cache_file, "w", encoding="utf-8") as f:
#                 json.dump(self.cache, f, indent=4)
#         except Exception as e:
#             print(f"[Cache Save Error] {e}")

#     def _normalize_cached_value(self, cached_value):
#         """Normalize old and new cache formats into a reward-range dictionary."""
#         if isinstance(cached_value, dict):
#             lower_bound = float(cached_value.get("min", 0.0))
#             upper_bound = float(cached_value.get("max", lower_bound))
#             thought_process = cached_value.get("thought_process", "cached reward range")
#             return {
#                 "min": lower_bound,
#                 "max": upper_bound,
#                 "thought_process": thought_process,
#             }

#         reward_value = float(cached_value)
#         return {
#             "min": reward_value,
#             "max": reward_value,
#             "thought_process": "cached point reward converted to range",
#         }

#     def get_reward_range(self, current_info, prev_info):
#         """
#         Query the OpenAI API for a bounded shaping reward range for the current transition.
#         """
#         curr_pos = current_info["agent_pos"]
#         prev_pos = prev_info["agent_pos"]
#         has_key = current_info["has_key"]
#         target = current_info["exit_pos"] if has_key else current_info["key_pos"]

#         # Use a transition signature as the cache key.
#         cache_key = f"prev:{prev_pos}_curr:{curr_pos}_key:{has_key}"

#         if cache_key in self.cache:
#             self.cache_hit_count += 1
#             if self.cache_hit_count % 1000 == 0:
#                 print(f"[Cache Hit] Reused {self.cache_hit_count} cached rewards so far.")
#             return self._normalize_cached_value(self.cache[cache_key])

#         self.api_call_count += 1
#         start_time = time.time()
#         map_string = current_info.get("global_map_string", "")

#         system_prompt = (
#             "You are an expert reinforcement learning reward shaper. "
#             "Your task is to observe a 2D maze and provide a strictly bounded shaping reward "
#             "range for the agent."
#         )
# #         user_prompt = f"""
# # Here is the current Global Map State:
# # {map_string}

# # Legend: 'A' = Agent, 'K' = Key, 'E' = Exit, 'S' = Start, '#' = Wall, '.' = Path.
# # Current Goal: Reach {'Exit (E)' if has_key else 'Key (K)'} at {target}.

# # Agent Context:
# # - Previous Position (row, col): {prev_pos}
# # - Current Position (row, col): {curr_pos}
# # - Has collected Key?: {has_key}

# # Instructions:
# # 1. Calculate the Manhattan distance from the previous position to the current goal.
# # 2. Calculate the Manhattan distance from the current position to the current goal.
# # 3. Compare them to determine whether the agent moved closer, hit a wall, or moved farther away.
# # 4. Assign a reward interval between {self.min_reward} and {self.max_reward}.
# # 5. The interval should be narrow when confidence is high and wider when confidence is lower.
# # 6. Ensure reward_lower_bound <= reward_upper_bound.

# # You must output only a valid JSON object in the following format:
# # {{
# #   "thought_process": "brief explanation...",
# #   "reward_lower_bound": <float>,
# #   "reward_upper_bound": <float>
# # }}
# # """
#         user_prompt = f"""
# You are an expert Reinforcement Learning (RL) reward-shaping AI. Your task is to evaluate an agent's recent move in a maze and assign a reward interval based on its progress toward the current goal.

# ### Environmental Context
# Global Map State:
# {map_string}

# Legend: 'A' = Agent, 'K' = Key, 'E' = Exit, 'S' = Start, '#' = Wall, '.' = Path.
# Current Goal: Reach {'Exit (E)' if has_key else 'Key (K)'} located at {target}.

# ### Agent State
# - Previous Position (row, col): {prev_pos}
# - Current Position (row, col): {curr_pos}
# - Has collected Key?: {has_key}

# ### Evaluation Instructions
# Follow these steps exactly to determine the reward interval, which must be strictly between {self.min_reward} and {self.max_reward}.

# 1. Coordinate Extraction: Identify the exact (row, col) coordinates for the Previous Position, Current Position, and Target.
# 2. Distance Calculation (Previous): Calculate the Manhattan distance from the Previous Position to the Target using the formula: |prev_row - target_row| + |prev_col - target_col|.
# 3. Distance Calculation (Current): Calculate the Manhattan distance from the Current Position to the Target using the same formula.
# 4. Move Assessment: Compare the distances and positions to classify the move:
#    - Goal Reached: curr_pos == target.
#    - Hit a Wall / Stuck: curr_pos == prev_pos.
#    - Moved Closer: Current distance < Previous distance.
#    - Moved Farther: Current distance > Previous distance.
# 5. Reward & Confidence Interval Assignment:
#    - Assign a positive baseline for moving closer, a negative baseline for moving farther, and a severe penalty for hitting a wall.
#    - Set the interval width based on confidence: Use a **narrow interval** (high confidence) if the move's value is absolute (e.g., reaching the goal or hitting a wall). Use a **wider interval** (lower confidence) if the long-term value is uncertain (e.g., moving closer, but potentially entering a dead-end path based on the map layout). 

# ### Output Format
# You must output ONLY a valid JSON object. Do not include markdown formatting like ```json.
# {{
#   "thought_process": "1. Target coords are... 2. Prev dist = |x-x| + |y-y| = ... 3. Curr dist = ... 4. Move Assessment: ... 5. Confidence logic: ...",
#   "reward_lower_bound": <float>,
#   "reward_upper_bound": <float>
# }}
# """
#         try:
#             print(f"[API Call {self.api_call_count}] Requesting reward for transition {cache_key}...")
#             response = self.client.chat.completions.create(
#                 model=self.model_name,
#                 response_format={"type": "json_object"},
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt},
#                 ],
#                 temperature=0.0,
#             )

#             content = response.choices[0].message.content
#             result = json.loads(content)
#             reward_range = {
#                 "min": float(result.get("reward_lower_bound", 0.0)),
#                 "max": float(result.get("reward_upper_bound", 0.0)),
#                 "thought_process": result.get("thought_process", ""),
#             }

#             self.cache[cache_key] = reward_range
#             self._save_cache()

#             latency = time.time() - start_time
#             print(
#                 f"[API Response] Reward range: "
#                 f"[{reward_range['min']}, {reward_range['max']}] "
#                 f"(latency: {latency:.2f}s)"
#             )

#             return reward_range

#         except Exception as e:
#             print(f"[LLM API Error] {e}")
#             return {
#                 "min": 0.0,
#                 "max": 0.0,
#                 "thought_process": "fallback after API error",
#             }

import json
import os
import time
from collections import deque
import hashlib
from pathlib import Path

from openai import OpenAI

from reward_config import LLM_REWARD_RANGE_CONFIG


def _load_env_file(dotenv_path: Path) -> None:
    """Best-effort .env loader (no external dependency).

    Only sets keys that are not already present in the environment.
    Supports lines like KEY=VALUE, optional quotes, and comments (# ...).
    """
    try:
        if not dotenv_path.exists() or not dotenv_path.is_file():
            return

        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if "#" in line:
                # Allow inline comments.
                line = line.split("#", 1)[0].strip()
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        # Never hard-fail training due to .env parsing.
        return


def _try_load_dotenv() -> None:
    """Attempt to load environment variables from a .env file.

    Prefers python-dotenv when installed; otherwise falls back to a minimal parser.
    Searches current working directory first, then the directory of this file.
    """
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
        load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
        return
    except Exception:
        pass

    _load_env_file(Path.cwd() / ".env")
    _load_env_file(Path(__file__).resolve().parent / ".env")


class OpenAILLM:
    def __init__(self, model_name="gpt-4o-mini", cache_file="llm_cache.json"):
        _try_load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Set it via a local .env file or export it before running training."
            )

        # Initialize the OpenAI client.
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.min_reward = LLM_REWARD_RANGE_CONFIG["step_min"]
        self.max_reward = LLM_REWARD_RANGE_CONFIG["step_max"]

        # Cache LLM outputs to avoid repeated API calls for identical transitions.
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.api_call_count = 0
        self.cache_hit_count = 0

    def _load_cache(self):
        """Load the local reward cache if it is available."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Cache Load Error] {e}")
        return {}

    def _save_cache(self):
        """Persist the reward cache to disk."""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=4)
        except Exception as e:
            print(f"[Cache Save Error] {e}")

    def _normalize_cached_value(self, cached_value):
        """Normalize old and new cache formats into a reward-range dictionary."""
        if isinstance(cached_value, dict):
            lower_bound = float(cached_value.get("min", 0.0))
            upper_bound = float(cached_value.get("max", lower_bound))
            # Fallback to thought_process for backward compatibility with old caches
            analysis = cached_value.get("state_analysis", cached_value.get("thought_process", "cached reward range"))
            return {
                "min": lower_bound,
                "max": upper_bound,
                "state_analysis": analysis,
            }

        reward_value = float(cached_value)
        return {
            "min": reward_value,
            "max": reward_value,
            "state_analysis": "cached point reward converted to range",
        }

    def _get_bfs_distance(self, grid, start, target):
        """Calculates the true path distance ignoring dynamic agents."""
        if start == target:
            return 0
            
        rows, cols = len(grid), len(grid[0])
        queue = deque([(start[0], start[1], 0)])
        visited = {tuple(start)}
        
        while queue:
            r, c, dist = queue.popleft()
            if (r, c) == tuple(target):
                return dist
                
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    if grid[nr][nc] != '#':  # Assuming '#' is the only hard wall
                        visited.add((nr, nc))
                        queue.append((nr, nc, dist + 1))
                        
        return float('inf') # Return infinity if unreachable

    def _get_local_sensation(self, grid, pos):
        """Extracts a 3x3 local grid around the agent to provide immediate topological context."""
        r, c = pos
        rows, cols = len(grid), len(grid[0])
        local_grid = []
        
        for dr in [-1, 0, 1]:
            row_str = ""
            for dc in [-1, 0, 1]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    row_str += grid[nr][nc]
                else:
                    row_str += "#"  # Treat out-of-bounds as walls
            local_grid.append(row_str)
            
        return "\n".join(local_grid)

    def get_reward_range(self, current_info, prev_info):
        """
        Query the OpenAI API for a bounded shaping reward range for the current transition.
        """
        # 1. Cast numpy integers to standard Python ints for clean string formatting and JSON safety
        curr_pos = tuple(int(x) for x in current_info["agent_pos"])
        prev_pos = tuple(int(x) for x in prev_info["agent_pos"])
        has_key = bool(current_info["has_key"])
        
        raw_target = current_info["exit_pos"] if has_key else current_info["key_pos"]
        target = tuple(int(x) for x in raw_target)

        # Include a stable maze-layout signature to prevent cross-maze cache collisions.
        # `global_map_string` changes every step (agent moves), so we prefer `maze_layout_string`.
        layout_string = current_info.get("maze_layout_string") or ""
        if not layout_string:
            layout_string = current_info.get("global_map_string", "")
            # Best-effort removal of agent marker to stabilize across steps.
            layout_string = layout_string.replace("A", ".")

        maze_hash = hashlib.sha1(layout_string.encode("utf-8")).hexdigest()[:10]

        # Use a transition signature as the cache key.
        cache_key = f"maze:{maze_hash}_prev:{prev_pos}_curr:{curr_pos}_key:{has_key}"

        if cache_key in self.cache:
            self.cache_hit_count += 1
            if self.cache_hit_count % 1000 == 0:
                print(f"[Cache Hit] Reused {self.cache_hit_count} cached rewards so far.")
            return self._normalize_cached_value(self.cache[cache_key])

        self.api_call_count += 1
        start_time = time.time()
        
        map_string = current_info.get("global_map_string", "")
        # Maze._get_global_state_string() uses spaces between tokens, so we must split on whitespace.
        grid = [row.split() for row in map_string.strip().splitlines() if row.strip()]
        
        # Calculate Neuro-Symbolic facts
        prev_bfs = self._get_bfs_distance(grid, prev_pos, target)
        curr_bfs = self._get_bfs_distance(grid, curr_pos, target)
        delta_distance = curr_bfs - prev_bfs
        
        local_sensation = self._get_local_sensation(grid, curr_pos)
        
        # Determine strict move result
        exit_pos = tuple(current_info["exit_pos"])
        
        if curr_pos == target:
            if has_key:
                move_result_string = "Ultimate Goal Reached (Exit)"
            else:
                move_result_string = "Sub-Goal Reached (Key Collected)"
        elif not has_key and curr_pos == exit_pos:
            # The agent stepped on the exit tile, but it doesn't have the key!
            move_result_string = "WARNING: Hit Exit WITHOUT Key. This is an invalid action."
        elif curr_pos == prev_pos:
            move_result_string = "Hit Wall / Stuck"
        else:
            move_result_string = "Moved Successfully"

        system_prompt = (
            "You are an expert reinforcement learning reward shaper. "
            "Your task is to observe programmatic evaluations of a 2D maze state and provide "
            "a strictly bounded shaping reward range for the agent based on absolute topological facts."
        )

        user_prompt = f"""
You are an expert Reinforcement Learning reward-shaping AI. Your goal is to evaluate a maze-solving agent's recent step and output a structured reward interval.

### Absolute Environment Facts (Pre-calculated)
- Goal: Reach {'Exit (E)' if has_key else 'Key (K)'}
- Move Result: {move_result_string}
- True Path Distance to Goal Before Move: {prev_bfs} steps
- True Path Distance to Goal After Move: {curr_bfs} steps
- Distance Delta: {delta_distance} (Negative means closer along the optimal path, Positive means farther)

### Local Sensation (Immediate 3x3 Grid Around Agent)
{local_sensation}

### Global Map Topology
{map_string}

### Evaluation Heuristics
### Evaluation Heuristics
You must output a continuous reward interval `[reward_lower_bound, reward_upper_bound]` strictly between {self.min_reward} and {self.max_reward}.
Base your evaluation on the Distance Delta and the Local Sensation:

1. CRITICAL EVENTS: Reaching the Key (Sub-Goal) or reaching the Exit WITH the Key (Ultimate Goal) is a success. Maximize the interval.
2. PENALTIES: Hitting a wall, moving into a dead-end, or hitting the Exit WITHOUT the Key represents a wasted or invalid step. Penalize heavily.
3. PROGRESS: If Distance Delta is negative, the agent is moving along the optimal path toward its CURRENT target. Reward positively.
4. REGRESSION: If Distance Delta is positive, the agent is walking away from the optimal path. Apply a negative reward.
5. UNCERTAINTY WIDTH:
   - Use a tight interval (e.g., width of 0.1) for absolute events (walls, goal reached, hitting exit without key).
   - Use a wider interval (e.g., width of 0.5) if the agent is in an open area exploring.
   
### Output Format
Output ONLY a valid JSON object. No markdown formatting like ```json. Provide only a very brief, high-level summary of your reasoning in the state_analysis field, omitting intermediate step-by-step logic.
{{
  "state_analysis": "<Brief high-level summary of the state and distance delta>",
  "reward_lower_bound": <float>,
  "reward_upper_bound": <float>
}}
"""
        try:
            print(f"[API Call {self.api_call_count}] Requesting reward for transition {cache_key}...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )

            content = response.choices[0].message.content
            result = json.loads(content)
            reward_range = {
                "min": float(result.get("reward_lower_bound", 0.0)),
                "max": float(result.get("reward_upper_bound", 0.0)),
                "state_analysis": result.get("state_analysis", ""),
            }

            self.cache[cache_key] = reward_range
            self._save_cache()

            latency = time.time() - start_time
            print(
                f"[API Response] Reward range: "
                f"[{reward_range['min']}, {reward_range['max']}] "
                f"(latency: {latency:.2f}s)"
            )

            return reward_range

        except Exception as e:
            print(f"[LLM API Error] {e}")
            return {
                "min": 0.0,
                "max": 0.0,
                "state_analysis": "fallback after API error",
            }