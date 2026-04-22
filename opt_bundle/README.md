# Optimized Bundle: Experiment Design and Reproducibility Guide

This directory contains the optimized experiment bundle for the maze-navigation reinforcement learning project. It is intended to be self-contained and report-ready: it documents the task definition, model architecture, reward design, LLM prompt design, experiment matrix, execution workflow, generated artifacts, and the exact naming conventions used by the current runs.

The bundle studies a partially observable sequential-goal navigation problem. In every maze, the agent must first find and collect a key, then navigate to the exit. The main research question is whether different reward formulations, especially LLM-based reward shaping, improve learning efficiency and final policy quality under a fixed RL setup.

## 1. Research Objective

The experiments in `optimized_bundle` are designed around the following controlled comparison:

1. Keep the task, environment, observation space, action space, optimizer, seed policy, and training budget fixed.
2. Change only the reward composition and, in some runs, the LLM backend.
3. Compare learning dynamics and final evaluation behavior across reward settings.

The main ablation axis is `reward_mode`:

- `sparse`
- `dense`
- `llm`
- `sparse_dense`
- `dense_llm`
- `sparse_llm`
- `sparse_dense_llm`

The code supports two RL algorithms:

- `A2C`
- `REINFORCE`

The bundle also contains two LLM backends:

- `Qwen/Qwen2.5-7B-Instruct` via local `transformers` inference in [QwenLLM.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/QwenLLM.py)
- OpenAI `gpt-4o-mini` via API calls in [OpenAILLM.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/OpenAILLM.py)

In practice, the current artifact set shows that the recent large-scale LLM runs were executed through the OpenAI entry point [main_openai.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/main_openai.py), while [main.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/main.py) keeps the local-Qwen path available.

## 2. Task Definition

The environment is implemented in [Maze.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/Maze.py) as a `gymnasium.Env`.

### 2.1 Maze semantics

Cell encoding:

- `0`: free path
- `1`: wall
- `2`: start position
- `3`: exit
- `4`: key
- `5`: agent marker used only in observations and visualization

Episode objective:

1. Start at `S`
2. Reach `K`
3. Pick up the key
4. Reach `E`

The exit does not count as success unless the key has already been collected.

### 2.2 Observation space

The environment is partially observable. The agent only sees a local `3 x 3` window centered on itself:

- observation shape: `(3, 3)`
- observation dtype: `int8`
- out-of-bound area is padded as walls

This makes recurrence necessary because a single observation does not reveal the full maze layout.

### 2.3 Action space

The action space is discrete with 4 actions:

- `0`: up
- `1`: down
- `2`: left
- `3`: right

### 2.4 Episode termination

Each episode ends when either:

- `terminated=True`: the agent reaches the exit while holding the key
- `truncated=True`: the step count reaches `max_steps`

The default training limit in the current experiment scripts is `500` steps per episode.

## 3. Dataset Design

Maze generation is implemented in [MazeGenerator.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/MazeGenerator.py).

### 3.1 Generation algorithms

Two maze-generation procedures are alternated:

- DFS-based maze generation for long winding corridors
- Prim-based maze generation for more branching structures and shorter dead ends

After generating a base maze, the script performs additional random wall removal ("hole punching") to increase structural diversity. Start, exit, and key locations are then sampled from free cells. The maze is accepted only if it is solvable in the required order:

`start -> key -> exit`

### 3.2 Dataset format

Each maze entry is a JSON object with:

- `id`
- `size`
- `algo`
- `grid`

Example semantic fields:

- `id="train_81"`
- `size=15`
- `algo="dfs"` or `algo="prim"`
- `grid=<2D integer array>`

### 3.3 Available datasets in this bundle

The local files currently present are:

- [dataset/train.json](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/dataset/train.json)
- [dataset/val.json](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/dataset/val.json)
- [dataset/test.json](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/dataset/test.json)
- [dataset/train_15_17.json](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/dataset/train_15_17.json)
- [dataset/val_15_17.json](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/dataset/val_15_17.json)
- [dataset/test_15_17.json](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/dataset/test_15_17.json)

Observed dataset counts in the current workspace:

- `train.json`: 1000 mazes = 500 of size 9 + 500 of size 25
- `val.json`: 200 mazes = 100 of size 9 + 100 of size 25
- `test.json`: 200 mazes = 100 of size 9 + 100 of size 25
- `train_15_17.json`: 1000 mazes = 500 of size 15 + 500 of size 17
- `val_15_17.json`: 200 mazes = 100 of size 15 + 100 of size 17
- `test_15_17.json`: 200 mazes = 100 of size 15 + 100 of size 17

This design is useful because each training run filters to one maze size, so a single mixed-size file can still support controlled single-size experiments.

## 4. Policy Architecture

The models are implemented in [A2C.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/A2C.py) and [REINFORCE.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/REINFORCE.py).

### 4.1 Shared encoder

Both algorithms share the same state encoding design:

- symbolic `3 x 3` observation
- `nn.Embedding(num_embeddings=6, embedding_dim=8)`
- flattened embedded grid
- concatenation with a scalar `has_key` feature
- GRU recurrent state with hidden size `64`

This means the policy receives both:

- local spatial content
- phase information: before-key vs after-key

### 4.2 A2C

`A2C` uses:

- an actor head for action logits
- a critic head for state value
- entropy regularization

Training loss:

`loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy`

### 4.3 REINFORCE

`REINFORCE` uses:

- a policy head only
- normalized discounted returns
- entropy regularization

Training loss:

`loss = policy_loss - entropy_coef * entropy`

## 5. Reward Design

Reward logic is split across [RewardEngine.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/RewardEngine.py), [reward_config.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/reward_config.py), [reward_manager.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/reward_manager.py), and the OpenAI-specific manager inside [main_openai.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/main_openai.py).

At each step, the total training reward is assembled from up to three components:

- sparse reward
- dense hand-crafted reward
- LLM-derived reward

The component combination is controlled by `--reward_mode`.

### 5.1 Sparse reward

Sparse reward is the task-ground-truth reward and comes directly from the environment:

- key pickup: `+20.0`
- successful exit after key: `+50.0`

### 5.2 Dense reward

Dense reward parameters are defined in [reward_config.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/reward_config.py):

- step penalty: `-0.1`
- revisit penalty: `-0.05`
- progress scale: `0.25`
- before key: weight only key distance
- after key: weight only exit distance

Dense progress is not based on Manhattan distance. It uses exact BFS shortest-path distance maps computed over the true maze topology. This is important because it makes the dense shaping topologically correct even in mazes with walls, loops, and dead ends.

### 5.3 LLM reward

LLM shaping is range-based rather than point-based.

For an eligible transition, the LLM returns:

- `reward_lower_bound`
- `reward_upper_bound`
- `state_analysis`

Then `RewardEngine.sample_llm_reward(...)` samples a scalar inside that interval and enforces a per-episode reward budget.

Current LLM reward configuration:

- per-step lower bound: `-0.05`
- per-step upper bound: `0.5`
- episode-level absolute budget ratio: `0.49`
- sparse reference ceiling: `50.0`
- default LLM total absolute budget per episode: `50.0 * 0.49 = 24.5`

Special scaling in `llm`-only mode:

- reward scale multiplier: `5.0`
- budget scale multiplier: `5.0`
- key bonus: `+5.0`
- exit bonus: `+20.0`
- exit-without-key penalty: `-2.0`

This scaling exists because otherwise a pure-LLM reward signal would be much weaker than the task milestones.

### 5.4 Reward modes

Supported reward modes are:

- `sparse`
- `dense`
- `llm`
- `sparse_dense`
- `dense_llm`
- `sparse_llm`
- `sparse_dense_llm`

Default mode:

- `sparse_dense_llm`

## 6. LLM Prompt Design

The LLM reward prompt is the most important experimental component in the bundle. It is designed to evaluate one transition using pre-computed symbolic features instead of asking the model to parse the whole maze from scratch.

### 6.1 System prompt

Both OpenAI and Qwen backends use the same short system prompt:

```text
You are an expert reinforcement learning reward-shaping AI.
Evaluate one maze transition using the provided pre-calculated facts.
Return only a valid JSON object.
```

### 6.2 User prompt structure

The user prompt contains structured features extracted from the transition:

- current sub-goal: key or exit
- move result string
- true BFS distance before move
- true BFS distance after move
- distance delta
- move type
- whether the move entered a dead-end corridor
- open-neighbor count before move
- open-neighbor count after move

The prompt asks the model to output only JSON:

```json
{
  "state_analysis": "<Brief high-level summary of the state and distance delta>",
  "reward_lower_bound": <float>,
  "reward_upper_bound": <float>
}
```

### 6.3 Full prompt template currently implemented

The exact prompt template used by the current code is:

```text
You are an expert Reinforcement Learning reward-shaping AI. Your goal is to evaluate a maze-solving agent's recent step and output a structured reward interval.

### Absolute Environment Facts (Pre-calculated)
- Goal: Reach Exit (E) or Key (K), depending on whether the key has been collected
- Move Result: one of success / invalid / stuck / exit without key
- True Path Distance to Goal Before Move: <prev_bfs> steps
- True Path Distance to Goal After Move: <curr_bfs> steps
- Distance Delta: <delta_distance>
- Move Type: <goal_reached | invalid_exit_without_key | invalid_or_stuck | progress | regression | neutral_valid_move>
- Entered Dead-End Corridor: <true/false>
- Open Neighbors Before Move: <prev_degree>
- Open Neighbors After Move: <curr_degree>

### Evaluation Heuristics
1. CRITICAL EVENTS: Reaching the Key or reaching the Exit with the Key is a success. Maximize the interval.
2. PENALTIES: Hitting a wall, moving into a dead-end, or hitting the Exit without the Key should be penalized heavily.
3. PROGRESS: If the move decreases BFS distance to the current goal, reward positively.
4. NEUTRAL VALID MOVE: If the move is legal but does not improve shortest-path distance, keep reward near zero or slightly negative.
5. REGRESSION: If the move increases BFS distance, apply a negative reward.
6. UNCERTAINTY WIDTH:
   - Use a tight interval for absolute events.
   - Use a wider interval in open exploratory situations.

### Output Format
Output ONLY a valid JSON object. No markdown formatting. Provide only a very brief, high-level summary in state_analysis.
```

### 6.4 Why this prompt is efficient

The prompt does not require the model to solve the maze. The expensive reasoning is replaced by deterministic features computed in code:

- exact BFS distance
- dead-end detection
- branching factor
- phase-aware target selection

This reduces prompt ambiguity, improves cache reusability, and makes the LLM act more like a heuristic evaluator than a planner.

### 6.5 Prompt-versioning and caching

Caching is built into both LLM backends:

- OpenAI prompt version key: `v3_feature_cache_compact`
- Qwen prompt version key: `qwen_local_transformers_v3_feature_cache_compact`

The cache key includes:

- prompt version
- model name
- reward bounds
- `has_key`
- `move_type`
- previous and current BFS distance
- delta distance
- dead-end status
- previous and current branching factor

This means semantically equivalent transitions across many episodes can reuse one LLM judgment, which is critical for cost control and runtime speed.

## 7. Transition Feature Engineering for LLM Reward

The LLM is never given raw policy logits or hidden states. It only receives human-interpretable transition features.

### 7.1 Derived move labels

Move result classes:

- goal reached
- invalid exit without key
- hit wall or stuck
- moved successfully

Move type classes:

- `goal_reached`
- `invalid_exit_without_key`
- `invalid_or_stuck`
- `progress`
- `regression`
- `neutral_valid_move`

### 7.2 Dead-end detection

A move is marked as entering a dead-end corridor when:

- the move changed position
- the destination is not the target
- previous open degree was greater than 1
- current open degree is less than or equal to 1

This allows the LLM to distinguish exploratory branching from locally harmful narrowing moves.

### 7.3 Fallback policy

Certain events bypass the API/model call and use deterministic fallback intervals:

- goal reached
- invalid exit without key
- invalid or stuck move

This guarantees stable shaping for unambiguous transitions and reduces unnecessary LLM usage.

## 8. Training Protocol

Training entry points:

- [main.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/main.py): local-Qwen pipeline
- [main_openai.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/main_openai.py): OpenAI pipeline

### 8.1 Common training settings observed in this bundle

The current experiment scripts and artifact names show a consistent protocol:

- episodes: `4000`
- learning rate: `0.001`
- entropy coefficient: `0.05`
- max steps: `500`
- seed: `42`
- top success GIFs: `3`

### 8.2 Maze sampling policy

Each episode:

1. loads one maze sampled uniformly at random from the chosen size subset
2. resets environment and recurrent hidden state
3. rolls out until success or truncation
4. writes one CSV row with total reward and component totals

### 8.3 Logged training fields

The training CSV stores:

- episode index
- maze ID
- maze size
- steps
- total reward
- sparse reward sum
- dense reward sum
- LLM reward sum
- sum of lower bounds
- sum of upper bounds
- success flag
- loss
- algorithm
- seed
- run name
- reward mode

OpenAI runs additionally log:

- `LLM_Backend`
- `LLM_Model`

## 9. Evaluation Protocol

Evaluation is handled by:

- [test_agent.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/test_agent.py): single-maze evaluation
- [run_test.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/run_test.py): convenience wrapper
- [run_test_all.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/run_test_all.py): whole test subset of one size

Important evaluation rule:

- evaluation uses environment sparse reward only

Dense reward and LLM reward are training-time shaping signals. They are not recomputed at evaluation time. This is the correct experimental choice if the goal is to compare policies under a shared task objective.

Batch evaluation summary metrics:

- success rate
- average steps
- average sparse reward
- average steps on successful mazes

## 10. Optimization-Specific Engineering

This bundle is not just a copy of the original project. It includes runtime optimizations intended to preserve semantics while reducing overhead.

Notable optimizations documented in [OPTIMIZED_README.md](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/OPTIMIZED_README.md):

- GPU use when available
- text-map generation only when LLM reward is active
- BFS distance-map caching across repeated mazes
- compact LLM cache design
- lower logging overhead
- reduced trajectory-frame storage overhead
- resource-aware experiment runner

These changes matter because LLM-shaped runs are substantially more expensive than non-LLM runs.

## 11. Experiment Matrix

The scripts in this directory define the intended experiment grid.

### 11.1 9x9 OpenAI experiment scripts

[script9a2c.txt](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/script9a2c.txt) lists A2C runs on `dataset/train.json`, size `9`, with:

- `llm`
- `dense_llm`
- `sparse_llm`
- `sparse_dense_llm`

[script9rein.txt](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/script9rein.txt) lists REINFORCE runs on `dataset/train.json`, size `9`, with:

- `llm`
- `dense_llm`
- `sparse_llm`
- `sparse_dense_llm`

### 11.2 15x15 experiment script

[script15.txt](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/script15.txt) lists both A2C and REINFORCE runs on `dataset/train_15_17.json`, filtered to size `15`.

The intended conditions are:

- `sparse`
- `dense`
- `llm`
- `sparse_dense`
- `dense_llm`
- `sparse_llm`
- `sparse_dense_llm`

for both algorithms.

### 11.3 Resource-aware runner

[run_pending_experiments.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/run_pending_experiments.py) executes pending checklist commands and classifies jobs as:

- `llm`
- `non_llm`

Default scheduling policy:

- up to 4 non-LLM jobs in parallel
- 1 LLM job at a time

This design is sensible because LLM runs are the bottleneck in both latency and cost.

## 12. Current Artifact Status in This Workspace

Based on the files currently present in `logs/`, `checkpoints/`, `runner_logs/`, and the checklist scripts, the local experiment state on April 19, 2026 is:

### 12.1 Completed or partially completed logged runs

A2C:

- `15x15 sparse`: completed, 4000 episodes
- `15x15 dense`: completed, 4000 episodes
- `15x15 llm (OpenAI)`: completed, 4000 episodes
- `15x15 sparse_dense`: completed, 4000 episodes
- `15x15 dense_llm (OpenAI)`: completed, 4000 episodes
- `15x15 sparse_llm (OpenAI)`: partially completed, 1461 logged episodes
- `9x9 llm (OpenAI)`: one 8000-episode log exists
- `9x9 dense_llm (OpenAI)`: partially completed, 2200 logged episodes

REINFORCE:

- `15x15 sparse`: completed, 4000 episodes
- `15x15 dense`: completed, 4000 episodes
- `15x15 sparse_dense`: completed, 4000 episodes
- `9x9 llm (OpenAI)`: partially completed, 2063 logged episodes

### 12.2 Checkpoint availability

Checkpoints currently exist for:

- A2C `15x15`: `sparse`, `dense`, `llm`, `sparse_dense`, `dense_llm`
- A2C `9x9`: `llm`
- REINFORCE `15x15`: `sparse`, `dense`, `sparse_dense`

No final checkpoint is currently present for partially completed runs such as:

- A2C `15x15 sparse_llm`
- A2C `9x9 dense_llm`
- REINFORCE `9x9 llm`

### 12.3 Why this status section matters

For a course report, this distinction is important:

- a log file indicates training progress existed
- a checkpoint indicates a recoverable trained model exists
- a checklist line marked `[ ]` indicates the run was planned but not confirmed finished through the runner

## 13. Output Artifacts and Naming Convention

The output folders are:

- `logs/<algo>/`
- `checkpoints/<algo>/`
- `plots/<algo>/`
- `gifs/<algo>/`
- `runner_logs/`

Typical OpenAI run tag:

```text
<run_name>_<algo>_<reward_mode>_openai_<llm_model>_size<maze_size>_lr<lr>_seed<seed>_<timestamp>
```

Typical non-OpenAI run tag:

```text
<run_name>_<algo>_<reward_mode>_size<maze_size>_lr<lr>_seed<seed>_<timestamp>
```

This naming scheme makes it possible to recover the full training condition from the filename alone.

## 14. Reproducibility Specification

To reproduce the current experiment design, keep the following fixed unless a new experiment explicitly changes them:

- maze size
- dataset split file
- algorithm
- reward mode
- LLM backend and model
- learning rate
- entropy coefficient
- max steps
- episode count
- seed

Recommended core commands:

```bash
python main_openai.py --dataset dataset/train_15_17.json --algo A2C --cache_file llm_cache_15.json --reward_mode dense_llm --maze_size 15 --episodes 4000 --lr 0.001 --max_steps 500 --seed 42 --run_name full_exp_train_15_17_15x15
```

```bash
python main.py --dataset dataset/train_15_17.json --algo A2C --reward_mode sparse_dense_llm --maze_size 15 --episodes 4000 --lr 0.001 --max_steps 500 --seed 42 --run_name local_qwen_exp
```

```bash
python run_test_all.py --checkpoint checkpoints/A2C/<checkpoint>.pt --dataset dataset/test_15_17.json --maze_size 15 --deterministic
```

## 15. Reporting Recommendations

If this README is used as the basis for a final report, the clearest structure is:

1. Describe the partially observable two-stage maze task.
2. Explain why recurrence is necessary.
3. Introduce the seven reward modes as the main ablation axis.
4. Emphasize that dense shaping uses exact BFS distance.
5. Emphasize that LLM shaping uses structured transition features rather than raw maze parsing.
6. Report training curves and held-out batch evaluation separately.
7. Distinguish planned runs from completed checkpointed runs.

For tables, the most useful columns are:

- algorithm
- maze size
- reward mode
- LLM backend
- training episodes
- checkpoint available
- test success rate
- average test steps

## 16. Limitations

Several limitations should be stated explicitly in any serious write-up:

- evaluation uses sparse task reward only, so shaped reward totals are not directly comparable to test returns
- current training samples one maze per episode rather than curriculum scheduling
- the 9x9 OpenAI artifact set is incomplete in the present workspace
- some LLM runs appear interrupted before checkpoint save
- only one seed is consistently used in the current scripts
- the bundle compares reward formulations more strongly than it compares architectures

## 17. File Map

Core experiment files:

- [main.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/main.py)
- [main_openai.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/main_openai.py)
- [Maze.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/Maze.py)
- [MazeGenerator.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/MazeGenerator.py)
- [A2C.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/A2C.py)
- [REINFORCE.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/REINFORCE.py)
- [RewardEngine.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/RewardEngine.py)
- [reward_config.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/reward_config.py)
- [reward_manager.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/reward_manager.py)
- [OpenAILLM.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/OpenAILLM.py)
- [QwenLLM.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/QwenLLM.py)
- [run_pending_experiments.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/run_pending_experiments.py)
- [run_test.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/run_test.py)
- [run_test_all.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/run_test_all.py)
- [test_agent.py](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/test_agent.py)

Experiment-control files:

- [script9a2c.txt](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/script9a2c.txt)
- [script9rein.txt](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/script9rein.txt)
- [script15.txt](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/script15.txt)
- [OPTIMIZED_README.md](/d:/USC_COURSE/CSCI_544/MazeAgentRL/optimized_bundle/OPTIMIZED_README.md)

