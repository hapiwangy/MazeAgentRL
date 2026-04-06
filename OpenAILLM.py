import json
import os
import time

from openai import OpenAI

from reward_config import LLM_REWARD_RANGE_CONFIG


class OpenAILLM:
    def __init__(self, model_name="gpt-4o-mini", cache_file="llm_cache.json"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Please export it before running training.")

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
            thought_process = cached_value.get("thought_process", "cached reward range")
            return {
                "min": lower_bound,
                "max": upper_bound,
                "thought_process": thought_process,
            }

        reward_value = float(cached_value)
        return {
            "min": reward_value,
            "max": reward_value,
            "thought_process": "cached point reward converted to range",
        }

    def get_reward_range(self, current_info, prev_info):
        """
        Query the OpenAI API for a bounded shaping reward range for the current transition.
        """
        curr_pos = current_info["agent_pos"]
        prev_pos = prev_info["agent_pos"]
        has_key = current_info["has_key"]
        target = current_info["exit_pos"] if has_key else current_info["key_pos"]

        # Use a transition signature as the cache key.
        cache_key = f"prev:{prev_pos}_curr:{curr_pos}_key:{has_key}"

        if cache_key in self.cache:
            self.cache_hit_count += 1
            if self.cache_hit_count % 1000 == 0:
                print(f"[Cache Hit] Reused {self.cache_hit_count} cached rewards so far.")
            return self._normalize_cached_value(self.cache[cache_key])

        self.api_call_count += 1
        start_time = time.time()
        map_string = current_info.get("global_map_string", "")

        system_prompt = (
            "You are an expert reinforcement learning reward shaper. "
            "Your task is to observe a 2D maze and provide a strictly bounded shaping reward "
            "range for the agent."
        )
        user_prompt = f"""
Here is the current Global Map State:
{map_string}

Legend: 'A' = Agent, 'K' = Key, 'E' = Exit, 'S' = Start, '#' = Wall, '.' = Path.
Current Goal: Reach {'Exit (E)' if has_key else 'Key (K)'} at {target}.

Agent Context:
- Previous Position (row, col): {prev_pos}
- Current Position (row, col): {curr_pos}
- Has collected Key?: {has_key}

Instructions:
1. Calculate the Manhattan distance from the previous position to the current goal.
2. Calculate the Manhattan distance from the current position to the current goal.
3. Compare them to determine whether the agent moved closer, hit a wall, or moved farther away.
4. Assign a reward interval between {self.min_reward} and {self.max_reward}.
5. The interval should be narrow when confidence is high and wider when confidence is lower.
6. Ensure reward_lower_bound <= reward_upper_bound.

You must output only a valid JSON object in the following format:
{{
  "thought_process": "brief explanation...",
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
                "thought_process": result.get("thought_process", ""),
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
                "thought_process": "fallback after API error",
            }
