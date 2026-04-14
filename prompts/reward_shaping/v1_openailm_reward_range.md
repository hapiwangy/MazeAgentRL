# v1 OpenAILLM Reward Range Prompt

## Source

- Runtime source: `OpenAILLM.py`
- Current usage: `OpenAILLM.get_reward_range()`
- Prompt style: one fixed system prompt plus one formatted user prompt

## How This Version Is Generated

This prompt version is assembled directly in Python at runtime.

1. `system_prompt` is a fixed string.
2. `user_prompt` is built with an f-string.
3. The following runtime values are injected into the user prompt:
   - `{global_map_string}`
   - `{prev_pos}`
   - `{curr_pos}`
   - `{has_key}`
   - `{target}`
   - `{min_reward}`
   - `{max_reward}`
4. The goal text changes depending on whether the agent already has the key:
   - if `has_key == False`, the goal is `Key (K)`
   - if `has_key == True`, the goal is `Exit (E)`
5. The final request is sent as chat messages with:
   - one `system` message
   - one `user` message

## System Prompt

```text
You are an expert reinforcement learning reward shaper. Your task is to observe a 2D maze and provide a strictly bounded shaping reward range for the agent.
```

## User Prompt Template

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

## Notes

- This version reasons with Manhattan distance.
- The project README notes that the hand-designed dense reward uses BFS path distance instead.
- This file is archival only for now; the code does not load from this file yet.
