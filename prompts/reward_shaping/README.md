# Reward-Shaping Prompt Versions

This document integrates the two prompt designs used in this project and explains how they differ.

## 1. Why There Are Two Versions

The project started with a more direct maze-description prompt and later moved to a structured feature-based prompt.

- old version:
  - map-centric
  - relied on the LLM reading the maze text and reasoning about position changes
  - described in `v1_openailm_reward_range.md`
- current version:
  - feature-centric
  - gives the LLM pre-computed symbolic facts about one transition
  - implemented directly in `OpenAILLM.py` and `QwenLLM.py`

The current code path uses the new feature-based prompt. The old markdown file is kept as an archival reference.

## 2. Version A: Old Map-Based Prompt

Source:

- archival doc: `v1_openailm_reward_range.md`

Main idea:

- send the global maze text map
- show the previous and current position
- tell the model whether the key has already been collected
- ask the model to infer whether the move got closer to the current goal

Core characteristics:

- uses the full map string
- reasons mainly through Manhattan distance
- output field was `thought_process`
- asks the model to do more of the spatial reasoning itself

Archived system prompt:

```text
You are an expert reinforcement learning reward shaper. Your task is to observe a 2D maze and provide a strictly bounded shaping reward range for the agent.
```

Archived user prompt template:

```text
Here is the current Global Map State:
{{global_map_string}}

Legend: 'A' = Agent, 'K' = Key, 'E' = Exit, 'S' = Start, '#' = Wall, '.' = Path.
Current Goal: Reach {{goal_label}} at {{target}}.

Agent Context:
- Previous Position (row, col): {{prev_pos}}
- Current Position (row, col): {{curr_pos}}
- Has collected Key?: {{has_key}}

Instructions:
1. Calculate the Manhattan distance from the previous position to the current goal.
2. Calculate the Manhattan distance from the current position to the current goal.
3. Compare them to determine whether the agent moved closer, hit a wall, or moved farther away.
4. Assign a reward interval between {{min_reward}} and {{max_reward}}.
5. The interval should be narrow when confidence is high and wider when confidence is lower.
6. Ensure reward_lower_bound <= reward_upper_bound.

You must output only a valid JSON object in the following format:
{
  "thought_process": "brief explanation...",
  "reward_lower_bound": <float>,
  "reward_upper_bound": <float>
}
```

Main limitation:

- the LLM had to infer topology and progress from the raw map text, which is more ambiguous and less cache-friendly

## 3. Version B: Current Structured Feature Prompt

Live sources:

- `OpenAILLM.py`
- `QwenLLM.py`

Prompt version identifiers:

- OpenAI: `v3_feature_cache_compact`
- Qwen: `qwen_local_transformers_v3_feature_cache_compact`

Main idea:

- do the maze reasoning in code first
- send only a compact set of pre-computed transition facts
- ask the LLM to evaluate one step and output a bounded reward interval

This version gives the model the following inputs:

- current goal: key or exit
- move result string
- true BFS distance before the move
- true BFS distance after the move
- distance delta
- move type
- whether the move entered a dead-end corridor
- open-neighbor count before the move
- open-neighbor count after the move

Current system prompt:

```text
You are an expert reinforcement learning reward-shaping AI. Evaluate one maze transition using the provided pre-calculated facts. Return only a valid JSON object.
```

Current user prompt template:

```text
You are an expert Reinforcement Learning reward-shaping AI. Your goal is to evaluate a maze-solving agent's recent step and output a structured reward interval.

### Absolute Environment Facts (Pre-calculated)
- Goal: Reach {Exit (E) if has_key else Key (K)}
- Move Result: {move_result_string}
- True Path Distance to Goal Before Move: {prev_bfs} steps
- True Path Distance to Goal After Move: {curr_bfs} steps
- Distance Delta: {delta_distance} (Negative means closer along the optimal path, Positive means farther)
- Move Type: {move_type}
- Entered Dead-End Corridor: {entered_dead_end}
- Open Neighbors Before Move: {prev_degree}
- Open Neighbors After Move: {curr_degree}

### Evaluation Heuristics
You must output a continuous reward interval [reward_lower_bound, reward_upper_bound] strictly between {min_reward} and {max_reward}.
Base your evaluation on the Move Type, Distance Delta, dead-end status, and local branching factor:
1. CRITICAL EVENTS: Reaching the Key (Sub-Goal) or reaching the Exit WITH the Key (Ultimate Goal) is a success. Maximize the interval.
2. PENALTIES: Hitting a wall, moving into a dead-end, or hitting the Exit WITHOUT the Key represents a wasted or invalid step. Penalize heavily.
3. PROGRESS: If Move Type is progress and Distance Delta is negative, the agent is moving toward its CURRENT target. Reward positively.
4. NEUTRAL VALID MOVE: If Move Type is neutral_valid_move and Distance Delta is zero, the move is legal but gave no shortest-path progress. Keep the reward near zero or slightly negative.
5. REGRESSION: If Move Type is regression and Distance Delta is positive, the agent is walking away from the optimal path. Apply a negative reward.
6. UNCERTAINTY WIDTH:
   - Use a tight interval for absolute events.
   - Use a wider interval in open exploratory situations.

### Output Format
Output ONLY a valid JSON object. No markdown formatting. Provide only a very brief, high-level summary in state_analysis.
{
  "state_analysis": "<Brief high-level summary of the state and distance delta>",
  "reward_lower_bound": <float>,
  "reward_upper_bound": <float>
}
```

Note:

- the OpenAI and Qwen prompts are effectively the same design
- the Qwen version uses almost identical wording, with only very minor phrasing differences

## 4. What Changed Between Versions

Key differences:

- global map string -> structured transition features
- Manhattan distance -> BFS shortest-path distance
- coordinate-centric reasoning -> move-type and topology-aware reasoning
- `thought_process` -> `state_analysis`
- open-ended spatial reading -> compact cached evaluator behavior

The redesign moves more deterministic logic into Python:

- BFS distance computation
- dead-end detection
- branching-factor counting
- move-result classification
- move-type classification

That makes the LLM operate more like a bounded heuristic judge than a planner.

## 5. Why The New Prompt Is Better For This Project

The feature-based prompt is a better fit for reward shaping because it:

- reduces ambiguity
- lowers token usage
- improves cache reuse across repeated transition patterns
- aligns the LLM reward with true maze topology
- makes the reward output easier to bound and debug

This is why the README and the current codebase treat the new prompt as the default experiment design, while the old prompt is preserved mainly for prompt-comparison experiments and documentation.
