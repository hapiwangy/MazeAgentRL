import atexit
import hashlib
import json
import os
import time
from collections import deque
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reward_config import LLM_REWARD_RANGE_CONFIG


def _load_env_file(dotenv_path: Path) -> None:
    """Best-effort .env loader without extra dependencies."""
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
    """Load .env from cwd or the current file directory."""
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
        load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
        return
    except Exception:
        pass

    _load_env_file(Path.cwd() / ".env")
    _load_env_file(Path(__file__).resolve().parent / ".env")


class QwenLLM:
    PROMPT_VERSION = "qwen_local_transformers_v1"

    def __init__(
        self,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        cache_file="llm_cache_qwen.json",
        save_every=64,
        max_new_tokens=180,
    ):
        _try_load_dotenv()
        self.model_name = model_name
        self.min_reward = float(LLM_REWARD_RANGE_CONFIG["step_min"])
        self.max_reward = float(LLM_REWARD_RANGE_CONFIG["step_max"])
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.save_every = int(save_every)
        self.max_new_tokens = int(max_new_tokens)
        self.pending_cache_writes = 0
        self.api_call_count = 0
        self.cache_hit_count = 0
        atexit.register(self._flush_cache_on_exit)

        print(f"Loading local Qwen model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        if torch.cuda.is_available():
            torch_dtype = torch.float16
            model_kwargs = {
                "torch_dtype": torch_dtype,
                "device_map": "auto",
                "trust_remote_code": True,
            }
        else:
            torch_dtype = torch.float32
            model_kwargs = {
                "torch_dtype": torch_dtype,
                "trust_remote_code": True,
            }

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.model.eval()

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
        local_sensation,
        map_string,
    ):
        return f"""
You are an expert Reinforcement Learning reward-shaping AI. Your goal is to evaluate a maze-solving agent's recent step and output a structured reward interval in JSON.

### Absolute Environment Facts (Pre-calculated)
- Goal: Reach {'Exit (E)' if has_key else 'Key (K)'}
- Move Result: {move_result_string}
- True Path Distance to Goal Before Move: {prev_bfs} steps
- True Path Distance to Goal After Move: {curr_bfs} steps
- Distance Delta: {delta_distance} (Negative means closer along the optimal path, Positive means farther)
- Move Type: {move_type}
- Entered Dead-End Corridor: {entered_dead_end}

### Local Sensation (Immediate 3x3 Grid Around Agent)
{local_sensation}

### Global Map Topology
{map_string}

### Evaluation Heuristics
You must output a continuous reward interval [reward_lower_bound, reward_upper_bound] strictly between {self.min_reward} and {self.max_reward}.
Base your evaluation on the Move Type, Distance Delta, and the Local Sensation:
1. CRITICAL EVENTS: Reaching the Key (Sub-Goal) or reaching the Exit WITH the Key (Ultimate Goal) is a success. Maximize the interval.
2. PENALTIES: Hitting a wall, moving into a dead-end, or hitting the Exit WITHOUT the Key represents a wasted or invalid step. Penalize heavily.
3. PROGRESS: If Move Type is progress and Distance Delta is negative, the agent is moving toward its CURRENT target. Reward positively.
4. NEUTRAL VALID MOVE: If Move Type is neutral_valid_move and Distance Delta is zero, the move is legal but gave no shortest-path progress. Keep the reward near zero or slightly negative.
5. REGRESSION: If Move Type is regression and Distance Delta is positive, the agent is walking away from the optimal path. Apply a negative reward.
6. UNCERTAINTY WIDTH:
   - Use a tight interval for absolute events (walls, goal reached, hitting exit without key).
   - Use a wider interval if the agent is in an open area exploring.

### Output Format
Output ONLY a valid JSON object. No markdown formatting. Provide only a very brief, high-level summary of your reasoning in the state_analysis field.
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
            return {
                "min": lower_bound,
                "max": upper_bound,
                "state_analysis": state_analysis,
            }

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

    def _get_bfs_distance(self, grid, start, target):
        if start == target:
            return 0

        rows, cols = len(grid), len(grid[0])
        queue = deque([(start[0], start[1], 0)])
        visited = {tuple(start)}

        while queue:
            row, col, dist = queue.popleft()
            if (row, col) == tuple(target):
                return dist

            for delta_row, delta_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_row, next_col = row + delta_row, col + delta_col
                if 0 <= next_row < rows and 0 <= next_col < cols and (next_row, next_col) not in visited:
                    if grid[next_row][next_col] != "#":
                        visited.add((next_row, next_col))
                        queue.append((next_row, next_col, dist + 1))

        return 10**9

    def _get_local_sensation(self, grid, pos):
        row, col = pos
        rows, cols = len(grid), len(grid[0])
        local_grid = []

        for delta_row in [-1, 0, 1]:
            current_row = []
            for delta_col in [-1, 0, 1]:
                next_row, next_col = row + delta_row, col + delta_col
                if 0 <= next_row < rows and 0 <= next_col < cols:
                    current_row.append(grid[next_row][next_col])
                else:
                    current_row.append("#")
            local_grid.append("".join(current_row))

        return "\n".join(local_grid)

    def _count_open_neighbors(self, grid, pos):
        row, col = pos
        rows, cols = len(grid), len(grid[0])
        open_neighbors = 0
        for delta_row, delta_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_row, next_col = row + delta_row, col + delta_col
            if 0 <= next_row < rows and 0 <= next_col < cols and grid[next_row][next_col] != "#":
                open_neighbors += 1
        return open_neighbors

    def _entered_dead_end(self, grid, prev_pos, curr_pos, target):
        if curr_pos == prev_pos or curr_pos == target:
            return False
        prev_degree = self._count_open_neighbors(grid, prev_pos)
        curr_degree = self._count_open_neighbors(grid, curr_pos)
        return curr_degree <= 1 and prev_degree > 1

    def _build_cache_key(self, current_info, prev_pos, curr_pos, has_key):
        layout_string = current_info.get("maze_layout_string") or ""
        if not layout_string:
            layout_string = current_info.get("global_map_string", "").replace("A", ".")
        maze_hash = hashlib.sha1(layout_string.encode("utf-8")).hexdigest()[:10]
        return (
            f"v:{self.PROMPT_VERSION}|model:{self.model_name}|"
            f"range:{self.min_reward}:{self.max_reward}|maze:{maze_hash}|"
            f"prev:{prev_pos}|curr:{curr_pos}|key:{has_key}"
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

    def _generate_json_response(self, system_prompt, user_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer([text], return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.2,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return self._parse_json_response(response_text)

    def _parse_json_response(self, response_text):
        cleaned_text = response_text.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            pass

        start = cleaned_text.find("{")
        end = cleaned_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = cleaned_text[start:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Failed to parse model output as JSON: {cleaned_text}")

    def get_reward_range(self, current_info, prev_info):
        curr_pos = tuple(int(x) for x in current_info["agent_pos"])
        prev_pos = tuple(int(x) for x in prev_info["agent_pos"])
        has_key = bool(current_info["has_key"])
        target = tuple(int(x) for x in (current_info["exit_pos"] if has_key else current_info["key_pos"]))
        exit_pos = tuple(int(x) for x in current_info["exit_pos"])

        cache_key = self._build_cache_key(current_info, prev_pos, curr_pos, has_key)
        if cache_key in self.cache:
            self.cache_hit_count += 1
            if self.cache_hit_count % 1000 == 0:
                print(f"[Cache Hit] Reused {self.cache_hit_count} cached rewards so far.")
            return self._normalize_cached_value(self.cache[cache_key])

        map_string = current_info.get("global_map_string", "")
        grid = [row.split() for row in map_string.strip().splitlines() if row.strip()]
        if not grid:
            fallback = self._fallback_reward_range("invalid_or_stuck", False)
            self._cache_set(cache_key, fallback)
            return fallback

        prev_bfs = self._get_bfs_distance(grid, prev_pos, target)
        curr_bfs = self._get_bfs_distance(grid, curr_pos, target)
        delta_distance = curr_bfs - prev_bfs
        local_sensation = self._get_local_sensation(grid, curr_pos)
        move_result_string = self._classify_move_result(curr_pos, prev_pos, target, exit_pos, has_key)
        move_type = self._classify_move_type(
            curr_pos,
            prev_pos,
            target,
            exit_pos,
            has_key,
            delta_distance,
        )
        entered_dead_end = self._entered_dead_end(grid, prev_pos, curr_pos, target)

        if move_type in {"goal_reached", "invalid_exit_without_key", "invalid_or_stuck"}:
            reward_range = self._fallback_reward_range(move_type, entered_dead_end=False)
            self._cache_set(cache_key, reward_range)
            return reward_range

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            has_key=has_key,
            move_result_string=move_result_string,
            prev_bfs=prev_bfs,
            curr_bfs=curr_bfs,
            delta_distance=delta_distance,
            move_type=move_type,
            entered_dead_end=entered_dead_end,
            local_sensation=local_sensation,
            map_string=map_string,
        )

        self.api_call_count += 1
        start_time = time.time()

        try:
            print(f"[API Call {self.api_call_count}] Requesting reward for transition {cache_key}...")
            result = self._generate_json_response(system_prompt, user_prompt)
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

            latency = time.time() - start_time
            print(
                f"[API Response] Reward range: "
                f"[{reward_range['min']}, {reward_range['max']}] "
                f"(latency: {latency:.2f}s)"
            )
            return reward_range
        except Exception as exc:
            print(f"[LLM API Error] {exc}")
            fallback = self._fallback_reward_range(move_type, entered_dead_end)
            fallback["min"], fallback["max"] = self._sanitize_reward_range(
                fallback["min"],
                fallback["max"],
            )
            self._cache_set(cache_key, fallback)
            return fallback
