# MazeAgentRL

MazeAgentRL is a reinforcement learning project for partially observable maze navigation.

Current experiment focus:

- reward-combination ablation across the same RL task
- analysis of LLM-based reward shaping inside the same training pipeline

In each maze, the agent must:

1. find the key,
2. pick up the key,
3. reach the exit.

The project currently includes:

- two RL algorithms: `A2C` and `REINFORCE`
- sparse reward from the environment
- hand-designed dense reward shaping
- optional LLM-based reward shaping
- maze dataset generation
- dataset visualization
- training logs, checkpoints, plots, and GIFs
- single-maze evaluation
- full test-set batch evaluation

## 1. Current Experiment Design

The current experimental design is centered on comparing reward strategies under a fixed maze-navigation setting.

Primary reward-combination comparison:

1. `sparse`
2. `dense`
3. `llm`
4. `sparse_dense`
5. `dense_llm`
6. `sparse_llm`
7. `sparse_dense_llm`

Recommended comparison protocol:

- keep the maze task, dataset split, algorithm, maze size, and training budget fixed
- vary only `--reward_mode`
- compare both training behavior and held-out evaluation results

Useful outputs already supported by the repository:

- training CSV logs in `logs/<algo>/`
- learning curves in `plots/<algo>/`
- per-maze and summary evaluation CSVs from `run_test_all.py`
- qualitative GIF trajectories in `gifs/<algo>/`

The secondary experiment direction in `todo.md` is comparing different LLM reward designs. The current repository only implements one LLM reward generator in `OpenAILLM.py`, so that direction is not yet a full multi-method benchmark in code.

## 2. Project Structure

Core files:

- `main.py`: training entry point
- `test_agent.py`: evaluate one maze with a trained checkpoint
- `run_test.py`: wrapper for single-maze evaluation
- `run_test_all.py`: evaluate a checkpoint on all mazes of one size in a dataset
- `Maze.py`: Gymnasium maze environment
- `A2C.py`: A2C network and agent
- `REINFORCE.py`: REINFORCE network and agent
- `RewardEngine.py`: sparse / dense / bounded LLM reward logic
- `reward_manager.py`: reward pipeline that combines reward components
- `reward_config.py`: reward constants and valid reward modes
- `OpenAILLM.py`: OpenAI-based reward range generator with local cache
- `MazeGenerator.py`: generate train / validation / test maze datasets
- `inspect_dataset.py`: save sample maze images for inspection
- `BFS_solver.py`: shortest-path baseline analysis
- `utils.py`: seeding, output-directory helpers, plotting, and GIF export

Generated or commonly used folders:

- `dataset/`: generated maze datasets
- `dataset_images/`: PNG visualizations from `inspect_dataset.py`
- `logs/`: per-episode CSV training logs
- `checkpoints/`: saved model checkpoints
- `plots/`: training curve figures
- `gifs/`: training and evaluation trajectory GIFs
- `eval_results/`: evaluation CSV outputs
- `llm_cache.json`: cache of LLM reward responses

## 3. Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Current `requirements.txt` contains:

- `torch`
- `gymnasium`
- `numpy`
- `matplotlib`
- `pillow`
- `openai`

If you use any reward mode that includes `llm`, you must set `OPENAI_API_KEY` before training.

Recommended (local `.env` file in this folder):

1. Create a file named `.env` in the project folder (it is already in `.gitignore`).
2. Add:

```bash
OPENAI_API_KEY=your_api_key
```

The code will auto-load this `.env` at runtime.

Linux / macOS:

```bash
export OPENAI_API_KEY=your_api_key
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

Notes:

- LLM reward is only used during training.
- Evaluation scripts do not call the LLM.

## 4. Problem Setting

### Partial observability

The environment is partially observable. The agent only receives a local `3 x 3` observation centered on itself.

Observation tile encoding:

- `0`: path
- `1`: wall
- `2`: start
- `3`: exit
- `4`: key
- `5`: agent

Action encoding:

- `0`: up
- `1`: down
- `2`: left
- `3`: right

Episode ending conditions:

- `terminated`: the agent reaches the exit after obtaining the key
- `truncated`: the agent reaches `max_steps`

### Environment reward

`MazeEnv` itself provides sparse reward only:

- `+20.0` when the key is picked up
- `+50.0` when the agent reaches the exit after picking up the key

During training, `main.py` may add dense reward and LLM reward on top of this sparse reward through `RewardManager`.

## 5. Dataset Generation

Generate the default datasets:

```bash
python MazeGenerator.py
```

This script does two things:

1. prints one sample `9x9` maze in the terminal
2. builds:
   - `dataset/train.json`
   - `dataset/val.json`
   - `dataset/test.json`

Default split sizes in the current code:

- train: `1000`
- val: `200`
- test: `200`

Current generation behavior:

- maze sizes are `9` and `25`
- the first half of each split is `9x9`, the second half is `25x25`
- generation alternates between `dfs` and `prim`
- random seeds are fixed in the script for reproducibility

Each maze record currently has these fields:

- `id`: maze identifier such as `train_0`
- `size`: maze size, currently `9` or `25`
- `algo`: generator type, `dfs` or `prim`
- `grid`: 2D maze grid

Grid encoding:

- `0`: path
- `1`: wall
- `2`: start
- `3`: exit
- `4`: key

The generator validates solvability with the required order:

`start -> key -> exit`

## 6. Dataset Inspection

Visualize random samples from the generated dataset:

```bash
python inspect_dataset.py
```

The current script saves:

- 5 random mazes from `dataset/train.json` to `dataset_images/train/`
- 3 random mazes from `dataset/val.json` to `dataset_images/val/`

Each maze is exported as a PNG file named by maze id.

## 7. Reward Design

Training reward is computed as:

```text
total_step_reward = selected combination of sparse / dense / llm components
```

The exact combination depends on `--reward_mode`.

### Supported reward modes

The current valid `--reward_mode` choices are:

- `sparse`
- `dense`
- `llm`
- `sparse_dense`
- `dense_llm`
- `sparse_llm`
- `sparse_dense_llm`

Current default:

- `sparse_dense_llm`

These seven modes are the main ablation axis in the current experiment design.

### Sparse reward

Defined by `MazeEnv` and reused in `RewardEngine`:

- key pickup: `+20.0`
- successful exit: `+50.0`

### Dense reward

Defined in `RewardEngine.py`.

It currently includes:

- step penalty: `-0.1`
- revisit penalty: `-0.05`
- progress bonus scaled by `0.25`

Important detail:

- progress is computed from BFS path-distance maps, not Manhattan distance
- before picking up the key, the weighted target is mainly the key
- after picking up the key, the target switches to the exit
- when the key is picked up, the visited set is cleared so the post-key phase starts fresh

### LLM reward

Defined by `OpenAILLM.py` and `RewardEngine.py`.

Current behavior:

- enabled only when the selected reward mode contains `llm`
- `OpenAILLM` sends the full maze state, previous position, current position, and current goal to the OpenAI API
- the model returns a JSON reward range
- `RewardEngine` clips that range to configured bounds and samples one scalar reward
- per-step range is clipped to `[-0.05, 0.5]`
- episode-level absolute LLM reward budget is capped to `50.0 * 0.49 = 24.5`
- responses are cached in `llm_cache.json`

One implementation detail to be aware of:

- the prompt asks the LLM to reason with Manhattan distance, while the hand-designed dense reward uses BFS path distance

## 8. Experimental Protocol

To run the current reward-combination study, repeat training with the same settings and change only `--reward_mode`.

Example template:

```bash
python main.py --algo REINFORCE --maze_size 9 --dataset dataset/train.json --lr 0.001 --entropy_coef 0.05 --max_steps 500 --episodes 2000 --reward_mode sparse --run_name reward_ablation
```

Then rerun with:

- `dense`
- `llm`
- `sparse_dense`
- `dense_llm`
- `sparse_llm`
- `sparse_dense_llm`

Recommended controlled variables:

- same algorithm
- same dataset
- same maze size
- same learning rate
- same entropy coefficient
- same episode budget
- same seed or same seed policy across runs

Recommended reporting metrics:

- training reward trend
- training success rate
- test success rate
- average evaluation steps
- trajectory quality from saved GIFs

For the planned "different LLM reward" experiment, the codebase would need extra prompt variants or additional LLM reward modules beyond the current `OpenAILLM.py` implementation.

## 9. Training

Train with `main.py`.

For the current study, training should be treated as a controlled ablation over reward modes.

### Example commands

Train `A2C` on `9x9` mazes:

```bash
python main.py --algo A2C --maze_size 9 --dataset dataset/train.json --lr 0.001 --entropy_coef 0.05 --max_steps 500 --episodes 2000
```

Train `REINFORCE` on `9x9` mazes:

```bash
python main.py --algo REINFORCE --maze_size 9 --dataset dataset/train.json --lr 0.001 --entropy_coef 0.05 --max_steps 500 --episodes 2000
```

Train with an explicit reward mode and run name:

```bash
python main.py --algo REINFORCE --maze_size 9 --dataset dataset/train.json --lr 0.001 --entropy_coef 0.05 --max_steps 500 --episodes 2000 --reward_mode sparse_dense_llm --run_name exp_reward_compare
```

### Main arguments

- `--algo`: `A2C` or `REINFORCE`
- `--dataset`: training dataset path, default `dataset/train.json`
- `--lr`: learning rate, default `0.001`
- `--entropy_coef`: entropy regularization coefficient, default `0.05`
- `--max_steps`: max steps per episode, default `500`
- `--episodes`: number of training episodes, default `2000`
- `--maze_size`: maze size filter, choices `9` or `25`, default `9`
- `--seed`: random seed, default `42`
- `--run_name`: output tag, default `default`
- `--top_success_gifs`: number of best successful trajectories to export after training, default `3`
- `--reward_mode`: reward combination mode, default `sparse_dense_llm`

### What the training script actually does

- loads the dataset JSON
- filters mazes by `--maze_size`
- samples one maze uniformly at random per episode
- resets recurrent memory at the beginning of each episode
- logs one CSV row per episode
- saves one checkpoint at the end of training
- saves a learning-curve figure at the end of training
- saves one trajectory GIF every `500` episodes
- saves the top successful GIFs at the end, ranked by:
  - higher total reward first
  - fewer steps next
  - earlier episode index next

### Training outputs

For algorithm `<algo>`, outputs are written to:

- `logs/<algo>/`
- `checkpoints/<algo>/`
- `plots/<algo>/`
- `gifs/<algo>/`

The training CSV columns are:

- `Episode`
- `Maze_ID`
- `Maze_Size`
- `Steps`
- `Total_Reward`
- `Sparse_Reward`
- `Dense_Reward`
- `LLM_Reward`
- `LLM_Range_Min`
- `LLM_Range_Max`
- `Success`
- `Loss`
- `Algo`
- `Seed`
- `Run_Name`
- `Reward_Mode`

Checkpoint metadata currently includes:

- algorithm
- reward mode
- maze size
- number of episodes trained
- learning rate
- entropy coefficient
- max steps
- seed
- run name
- model weights

## 10. Single-Maze Evaluation

Use `test_agent.py` to evaluate one maze with a trained checkpoint.

### Important behavior

- evaluation uses the environment reward only, which is sparse reward
- dense reward and LLM reward are not recomputed during evaluation
- if `--maze_id` is omitted, one maze is chosen randomly from the selected size
- if `--maze_size`, `--max_steps`, or `--seed` are omitted, values are taken from the checkpoint when available

### Example commands

```bash
python test_agent.py --checkpoint checkpoints/REINFORCE/your_model.pt --dataset dataset/test.json --deterministic --save_gif
```

Evaluate a specific maze:

```bash
python test_agent.py --checkpoint checkpoints/REINFORCE/your_model.pt --dataset dataset/test.json --maze_id test_81 --deterministic --save_gif
```

Override maze size:

```bash
python test_agent.py --checkpoint checkpoints/REINFORCE/your_model.pt --dataset dataset/test.json --maze_size 9 --deterministic
```

### Arguments

- `--checkpoint`: required checkpoint path
- `--dataset`: evaluation dataset path, default `dataset/test.json`
- `--maze_size`: optional size filter, otherwise use checkpoint metadata
- `--maze_id`: optional maze id string
- `--max_steps`: optional max-step override, otherwise use checkpoint metadata
- `--deterministic`: use greedy argmax action selection
- `--save_gif`: save one evaluation GIF
- `--seed`: optional evaluation seed override, otherwise use checkpoint seed or `42`
- `--run_name`: output tag, default `eval`

### Outputs

Printed summary includes:

- checkpoint path
- algorithm
- checkpoint seed
- evaluation seed
- maze id
- maze size
- whether deterministic mode is enabled
- number of steps
- success or failure
- sparse reward sum

Saved files:

- one CSV under `eval_results/<algo>/`
- one GIF under `gifs/<algo>/` when `--save_gif` is enabled

## 11. `run_test.py` Wrapper

`run_test.py` is a convenience wrapper around `test_agent.py`.

Current behavior:

- defaults to `dataset/test.json`
- if `--checkpoint` is omitted, it finds the latest `.pt` file under `checkpoints/*/`
- forwards all selected arguments to `test_agent.py`

Example:

```bash
python run_test.py --deterministic --save_gif
```

Use a specific checkpoint:

```bash
python run_test.py --checkpoint checkpoints/REINFORCE/your_model.pt --maze_id test_81 --deterministic --save_gif
```

Supported arguments:

- `--checkpoint`
- `--dataset`
- `--maze_id`
- `--maze_size`
- `--max_steps`
- `--seed`
- `--run_name`
- `--deterministic`
- `--save_gif`

## 12. Batch Evaluation

Use `run_test_all.py` to evaluate one checkpoint on all mazes of the selected size in a dataset.

### Important behavior

- evaluation is filtered by one maze size only
- if `--maze_size` is omitted, the script uses the checkpoint's stored maze size
- evaluation reward is sparse reward only
- one detailed CSV and one summary CSV are always saved
- GIF export is optional and saves one GIF per maze

### Example commands

Basic batch evaluation:

```bash
python run_test_all.py --deterministic
```

Batch evaluation with GIFs:

```bash
python run_test_all.py --deterministic --save_gifs
```

Batch evaluation with explicit checkpoint:

```bash
python run_test_all.py --checkpoint checkpoints/REINFORCE/your_model.pt --maze_size 9 --deterministic
```

### Arguments

- `--checkpoint`: optional checkpoint path
- `--dataset`: evaluation dataset path, default `dataset/test.json`
- `--maze_size`: optional maze size filter, otherwise use checkpoint metadata
- `--max_steps`: optional max-step override, otherwise use checkpoint metadata
- `--seed`: optional evaluation seed override, otherwise use checkpoint seed or `42`
- `--run_name`: output tag, default `eval_test_all`
- `--deterministic`: use greedy argmax action selection
- `--save_gifs`: save one GIF for every evaluated maze

### Output files

Saved under `eval_results/<algo>/`:

- detail CSV: one row per maze
- summary CSV: aggregate metrics for the run

Summary metrics currently include:

- number of mazes
- success rate
- average steps
- average sparse reward
- average steps on successful mazes

This summary output is the main artifact for comparing the seven reward modes on the held-out test set.

## 13. GIF Visualization

Trajectory GIFs are generated by `utils.py`.

Current rendering behavior:

- path: white
- wall: black
- start: blue
- exit: green
- key: gold
- agent: red

GIF title text shows:

- episode or maze id label
- current step and total steps
- `Running` or `Finished`

Training GIF frequency:

- every `500` episodes during training
- plus the top successful episodes saved at the end

Evaluation GIF frequency:

- optional, controlled by `--save_gif` or `--save_gifs`

## 14. Model Details

Both policies use the same observation pipeline:

- the `3 x 3` symbolic observation is embedded with `nn.Embedding`
- the embedded grid is flattened
- a `GRU` stores recurrent memory across time steps

This recurrent design is important because the environment is partially observable.

### A2C

`A2C` contains:

- policy head
- value head
- entropy regularization during training

Loss structure:

- actor loss
- critic MSE loss
- entropy bonus

### REINFORCE

`REINFORCE` contains:

- policy head only
- normalized discounted returns
- entropy regularization during training

## 15. BFS Baseline

`BFS_solver.py` provides a shortest-path reference for the ordered objective:

`start -> key -> exit`

It returns:

- steps from start to key
- steps from key to exit
- total optimal steps

This is useful for error analysis and for comparing learned behavior with shortest valid paths.

## 16. Typical Workflow

Recommended workflow:

1. install dependencies
2. set `OPENAI_API_KEY` if you plan to use any `llm` reward mode
3. generate datasets with `python MazeGenerator.py`
4. inspect sample mazes with `python inspect_dataset.py`
5. train with `python main.py ...`
6. evaluate one maze with `python test_agent.py ...` or `python run_test.py ...`
7. evaluate the full test subset for one maze size with `python run_test_all.py ...`
8. analyze CSV logs, plots, and GIFs

## 17. Notes and Limitations

- `main.py` only trains on one maze size per run because it filters the dataset by `--maze_size`.
- `test_agent.py` and `run_test_all.py` report sparse environment reward, not dense reward and not LLM reward.
- Any reward mode containing `llm` requires a valid OpenAI API key during training.
- LLM reward calls can be slow and may increase training cost.
- `main.py` currently instantiates the LLM reward generator with model `gpt-4o-mini`.
- reward-combination comparison is fully supported, but multiple implemented LLM reward variants are not yet available in the repository.
- `llm_cache.json` may contain either older point-value entries or newer range-style entries; `OpenAILLM.py` normalizes both formats when loading.
- If you change the network architecture in `A2C.py` or `REINFORCE.py`, old checkpoints may fail to load.
- The current code stores only one checkpoint at the end of training, not intermediate checkpoints.

## 18. Quick Start

Train:

```bash
python main.py --algo REINFORCE --maze_size 9 --dataset dataset/train.json --lr 0.001 --entropy_coef 0.05 --max_steps 500 --episodes 2000 --reward_mode sparse_dense_llm --run_name exp1
```

Evaluate one maze:

```bash
python run_test.py --deterministic --save_gif
```

Evaluate the whole test subset of one maze size:

```bash
python run_test_all.py --deterministic
```
