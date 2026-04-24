import atexit
import json
import os
import time
from pathlib import Path

import numpy as np
from openai import OpenAI

from reward_config import LLM_REWARD_RANGE_CONFIG


def _load_env_file(dotenv_path: Path) -> None:
    try:
        if not dotenv_path.exists() or not dotenv_path.is_file():
            return

        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if "#" in line:
                line = line.split("#", 1)[0].strip()
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        return


def _try_load_dotenv() -> None:
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
    PROMPT_VERSION = "v3_feature_cache_compact"

    def __init__(self, model_name="gpt-4o-mini", cache_file="llm_cache.json", save_every=64, verbose=None, log_every=50):
        _try_load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Set it in a local .env file or export it before training.")

        cache_path = Path(cache_file)
        if not cache_path.is_absolute():
            cache_path = Path(__file__).resolve().parent / cache_path

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.min_reward = float(LLM_REWARD_RANGE_CONFIG["step_min"])
        self.max_reward = float(LLM_REWARD_RANGE_CONFIG["step_max"])
        self.cache_file = str(cache_path)
        self.cache = self._load_cache()
        self.save_every = int(save_every)
        self.verbose = bool(int(os.getenv("OPENAI_LLM_VERBOSE", "0"))) if verbose is None else bool(verbose)
        self.log_every = max(1, int(log_every))
        self.pending_cache_writes = 0
        self.api_call_count = 0
        self.cache_hit_count = 0
        atexit.register(self._flush_cache_on_exit)
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self):
        return (
            "You are an expert reinforcement learning reward-shaping AI. "
            "Evaluate one maze transition using the provided pre-calculated facts. "
            "Return only a valid JSON object."
        )

    def _build_user_prompt(
        self,
        has_key,
        move_result_string,
        prev_bfs,
        curr_bfs,
        delta_distance,
        move_type,
        entered_dead_end,
        prev_degree,
        curr_degree,
    ):
        return f"""
You are an expert Reinforcement Learning reward-shaping AI. Your goal is to evaluate a maze-solving agent's recent step and output a structured reward interval.

### Absolute Environment Facts (Pre-calculated)
- Goal: Reach {'Exit (E)' if has_key else 'Key (K)'}
- Move Result: {move_result_string}
- True Path Distance to Goal Before Move: {prev_bfs} steps
- True Path Distance to Goal After Move: {curr_bfs} steps
- Distance Delta: {delta_distance} (Negative means closer along the optimal path, Positive means farther)
- Move Type: {move_type}
- Entered Dead-End Corridor: {entered_dead_end}
- Open Neighbors Before Move: {prev_degree}
- Open Neighbors After Move: {curr_degree}

### Evaluation Heuristics
You must output a continuous reward interval [reward_lower_bound, reward_upper_bound] strictly between {self.min_reward} and {self.max_reward}.
Base your evaluation on the Move Type, Distance Delta, dead-end status, and local branching factor:
1. CRITICAL EVENTS: Reaching the Key (Sub-Goal) or reaching the Exit WITH the Key (Ultimate Goal) is a success. Maximize the interval.
2. PENALTIES: Hitting a wall, moving into a dead-end, or hitting the Exit WITHOUT the Key represents a wasted or invalid step. Penalize heavily.
3. PROGRESS: If Move Type is progress and Distance Delta is negative, the agent is moving toward its CURRENT target. Reward positively.
4. NEUTRAL VALID MOVE: If Move Type is neutral_valid_move and Distance Delta is zero, the move is legal but gave no shortest-path progress. Keep the reward near zero or slightly negative.
5. REGRESSION: If Move Type is regression and Distance Delta is positive, the agent is walking away from the optimal path. Apply a negative reward.
6. UNCERTAINTY WIDTH:
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

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as file:
                    return json.load(file)
            except Exception as exc:
                print(f"[Cache Load Error] {exc}")
        return {}

    def _save_cache(self):
        try:
            cache_dir = os.path.dirname(self.cache_file)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as file:
                json.dump(self.cache, file, indent=4)
            self.pending_cache_writes = 0
        except Exception as exc:
            print(f"[Cache Save Error] {exc}")

    def _flush_cache_on_exit(self):
        if self.pending_cache_writes > 0:
            self._save_cache()

    def _cache_set(self, cache_key, reward_range):
        self.cache[cache_key] = reward_range
        self.pending_cache_writes += 1
        if self.pending_cache_writes >= self.save_every:
            self._save_cache()

    def _normalize_cached_value(self, cached_value):
        if isinstance(cached_value, dict):
            lower_bound = float(cached_value.get("min", 0.0))
            upper_bound = float(cached_value.get("max", lower_bound))
            state_analysis = cached_value.get(
                "state_analysis",
                cached_value.get("thought_process", "cached reward range"),
            )
            lower_bound, upper_bound = self._sanitize_reward_range(lower_bound, upper_bound)
            return {"min": lower_bound, "max": upper_bound, "state_analysis": state_analysis}

        reward_value = float(cached_value)
        reward_value, reward_value = self._sanitize_reward_range(reward_value, reward_value)
        return {
            "min": reward_value,
            "max": reward_value,
            "state_analysis": "cached point reward converted to range",
        }

    def _sanitize_reward_range(self, lower_bound, upper_bound):
        eps = 1e-6
        lower = max(self.min_reward + eps, min(float(lower_bound), self.max_reward - eps))
        upper = max(self.min_reward + eps, min(float(upper_bound), self.max_reward - eps))
        if lower > upper:
            lower, upper = upper, lower
        return lower, upper

    def _count_open_neighbors(self, maze_grid, pos):
        row, col = pos
        rows, cols = maze_grid.shape
        open_neighbors = 0
        for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            next_row, next_col = row + delta_row, col + delta_col
            if 0 <= next_row < rows and 0 <= next_col < cols and maze_grid[next_row, next_col] != 1:
                open_neighbors += 1
        return open_neighbors

    def _entered_dead_end(self, maze_grid, prev_pos, curr_pos, target):
        if curr_pos == prev_pos or curr_pos == target:
            return False
        prev_degree = self._count_open_neighbors(maze_grid, prev_pos)
        curr_degree = self._count_open_neighbors(maze_grid, curr_pos)
        return curr_degree <= 1 and prev_degree > 1

    def _build_cache_key(self, has_key, prev_bfs, curr_bfs, delta_distance, move_type, entered_dead_end, prev_degree, curr_degree):
        return (
            f"v:{self.PROMPT_VERSION}|model:{self.model_name}|range:{self.min_reward}:{self.max_reward}|"
            f"key:{has_key}|move:{move_type}|prev_bfs:{prev_bfs}|curr_bfs:{curr_bfs}|"
            f"delta:{delta_distance}|dead_end:{entered_dead_end}|prev_deg:{prev_degree}|curr_deg:{curr_degree}"
        )

    def _classify_move_result(self, curr_pos, prev_pos, target, exit_pos, has_key):
        if curr_pos == target:
            return "Ultimate Goal Reached (Exit)" if has_key else "Sub-Goal Reached (Key Collected)"
        if not has_key and curr_pos == exit_pos:
            return "WARNING: Hit Exit WITHOUT Key. This is an invalid action."
        if curr_pos == prev_pos:
            return "Hit Wall / Stuck"
        return "Moved Successfully"

    def _classify_move_type(self, curr_pos, prev_pos, target, exit_pos, has_key, delta_distance):
        if curr_pos == target:
            return "goal_reached"
        if not has_key and curr_pos == exit_pos:
            return "invalid_exit_without_key"
        if curr_pos == prev_pos:
            return "invalid_or_stuck"
        if delta_distance < 0:
            return "progress"
        if delta_distance > 0:
            return "regression"
        return "neutral_valid_move"

    def _fallback_reward_range(self, move_type, entered_dead_end):
        if move_type == "goal_reached":
            return {
                "min": self.max_reward - 0.05,
                "max": self.max_reward - 0.01,
                "state_analysis": "The agent reached the current goal successfully.",
            }
        if move_type in {"invalid_exit_without_key", "invalid_or_stuck"}:
            return {
                "min": self.min_reward + 0.005,
                "max": self.min_reward + 0.03,
                "state_analysis": "The move was invalid or wasted and made no useful progress.",
            }
        if entered_dead_end:
            return {
                "min": self.min_reward + 0.008,
                "max": 0.0,
                "state_analysis": "The agent moved into a cramped dead-end area with low future value.",
            }
        if move_type == "progress":
            return {
                "min": 0.12,
                "max": 0.28,
                "state_analysis": "The agent moved closer to the current goal along the true path.",
            }
        if move_type == "regression":
            return {
                "min": self.min_reward + 0.01,
                "max": -0.005,
                "state_analysis": "The agent moved away from the current goal.",
            }
        return {
            "min": self.min_reward + 0.02,
            "max": 0.025,
            "state_analysis": "The move was legal but produced no shortest-path progress toward the current goal.",
        }

    def get_reward_range(self, current_info, prev_info):
        curr_pos = tuple(current_info["agent_pos"])
        prev_pos = tuple(prev_info["agent_pos"])
        has_key = bool(current_info["has_key"])
        target = tuple(current_info["exit_pos"] if has_key else current_info["key_pos"])
        exit_pos = tuple(current_info["exit_pos"])
        prev_bfs = prev_info["exit_distance"] if has_key else prev_info["key_distance"]
        curr_bfs = current_info["exit_distance"] if has_key else current_info["key_distance"]
        delta_distance = curr_bfs - prev_bfs
        move_result_string = self._classify_move_result(curr_pos, prev_pos, target, exit_pos, has_key)
        move_type = self._classify_move_type(curr_pos, prev_pos, target, exit_pos, has_key, delta_distance)
        maze_grid = np.asarray(current_info["maze_grid"])
        prev_degree = self._count_open_neighbors(maze_grid, prev_pos)
        curr_degree = self._count_open_neighbors(maze_grid, curr_pos)
        entered_dead_end = self._entered_dead_end(maze_grid, prev_pos, curr_pos, target)
        cache_key = self._build_cache_key(
            has_key=has_key,
            prev_bfs=prev_bfs,
            curr_bfs=curr_bfs,
            delta_distance=delta_distance,
            move_type=move_type,
            entered_dead_end=entered_dead_end,
            prev_degree=prev_degree,
            curr_degree=curr_degree,
        )
        if cache_key in self.cache:
            self.cache_hit_count += 1
            if self.verbose and self.cache_hit_count % 1000 == 0:
                print(f"[Cache Hit] Reused {self.cache_hit_count} cached rewards so far.")
            return self._normalize_cached_value(self.cache[cache_key])

        if move_type in {"goal_reached", "invalid_exit_without_key", "invalid_or_stuck"}:
            reward_range = self._fallback_reward_range(move_type, entered_dead_end=False)
            self._cache_set(cache_key, reward_range)
            return reward_range

        user_prompt = self._build_user_prompt(
            has_key=has_key,
            move_result_string=move_result_string,
            prev_bfs=prev_bfs,
            curr_bfs=curr_bfs,
            delta_distance=delta_distance,
            move_type=move_type,
            entered_dead_end=entered_dead_end,
            prev_degree=prev_degree,
            curr_degree=curr_degree,
        )

        self.api_call_count += 1
        start_time = time.perf_counter()

        try:
            if self.verbose:
                print(f"[API Call {self.api_call_count}] Requesting reward for transition {cache_key}...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            result = json.loads(response.choices[0].message.content)
            reward_min, reward_max = self._sanitize_reward_range(
                result.get("reward_lower_bound", 0.0),
                result.get("reward_upper_bound", 0.0),
            )
            reward_range = {
                "min": reward_min,
                "max": reward_max,
                "state_analysis": str(result.get("state_analysis", "")).strip(),
            }
            self._cache_set(cache_key, reward_range)
            latency = time.perf_counter() - start_time
            if self.verbose:
                print(f"[API Response] Reward range: [{reward_range['min']}, {reward_range['max']}] (latency: {latency:.2f}s)")
            elif self.api_call_count % self.log_every == 0:
                print(
                    f"[LLM] calls={self.api_call_count} cache_hits={self.cache_hit_count} "
                    f"last_latency={latency:.2f}s range=[{reward_range['min']}, {reward_range['max']}]"
                )
            return reward_range
        except Exception as exc:
            print(f"[LLM API Error] {exc}")
            fallback = self._fallback_reward_range(move_type, entered_dead_end)
            fallback["min"], fallback["max"] = self._sanitize_reward_range(fallback["min"], fallback["max"])
            self._cache_set(cache_key, fallback)
            return fallback
