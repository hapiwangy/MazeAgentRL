# MazeAgentRL

MazeAgentRL is a reinforcement learning project for a partially observable maze task with sequential goals:

1. find the key
2. pick it up
3. reach the exit

The repository includes:

- maze generation
- a `gymnasium` environment with `3 x 3` local observations
- recurrent `A2C` and `REINFORCE` agents
- sparse, dense, and LLM-based reward shaping
- local Qwen and OpenAI reward pipelines
- training logs, checkpoints, plots, and trajectory GIFs

## Task Setup

Maze cell encoding:

- `0`: free path
- `1`: wall
- `2`: start
- `3`: exit
- `4`: key
- `5`: agent marker used only in observations and visualization

The agent only sees a local `3 x 3` window centered on itself, plus a `has_key` flag that is passed into the policy. Reaching the exit without the key does not end the episode.

## Repository Layout

Core files:

- `Maze.py`: environment definition
- `MazeGenerator.py`: dataset generation
- `A2C.py`: recurrent actor-critic agent
- `REINFORCE.py`: recurrent policy-gradient agent
- `RewardEngine.py`: sparse reward, dense reward, BFS features, and LLM reward budgeting
- `reward_config.py`: reward constants and supported reward modes
- `reward_manager.py`: local-Qwen reward manager
- `OpenAILLM.py`: OpenAI reward backend
- `QwenLLM.py`: local `transformers` reward backend
- `main.py`: training entry point for local Qwen reward shaping
- `main_openai.py`: training entry point for OpenAI reward shaping
- `test_agent.py`: single-maze evaluation
- `run_test.py`: convenience wrapper for single-maze evaluation
- `run_test_all.py`: batch evaluation over one dataset split
- `utils.py`: plotting, GIF export, output-path helpers, and seeding
- `prompts/reward_shaping/README.md`: prompt-version notes

Generated folders are created automatically when needed:

- `dataset/`
- `logs/<algo>/`
- `checkpoints/<algo>/`
- `plots/<algo>/`
- `gifs/<algo>/`
- `eval_results/<algo>/`

## Reward Modes

Supported reward modes:

- `sparse`
- `dense`
- `llm`
- `sparse_dense`
- `dense_llm`
- `sparse_llm`
- `sparse_dense_llm`

Current reward configuration in `reward_config.py`:

- sparse reward:
  - `+20.0` for picking up the key
  - `+50.0` for reaching the exit after the key
- dense reward:
  - step penalty `-0.1`
  - revisit penalty `-0.05`
  - BFS progress shaping scaled by `0.25`
- LLM reward:
  - per-step interval bounded in `[-0.05, 0.5]`
  - episode-level absolute budget ratio `0.49`

The dense shaping uses BFS shortest-path distance over the true maze topology rather than Manhattan distance.

## Prompt Design

The live LLM prompt is feature-based. Instead of asking the model to parse the whole maze, the code computes symbolic transition features first and asks the LLM to score one step.

The current prompt logic lives in:

- `OpenAILLM.py`
- `QwenLLM.py`

Prompt-version notes and the archived older prompt are documented in:

- `prompts/reward_shaping/README.md`
- `prompts/reward_shaping/v1_openailm_reward_range.md`

## Setup

Recommended Python version:

- Python `3.10+`

Install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` includes:

- `torch`
- `gymnasium`
- `numpy`
- `matplotlib`
- `pillow`
- `openai`
- `python-dotenv`
- `transformers`
- `accelerate`

### OpenAI setup

For `main_openai.py`, create a local `.env` file in the repo root:

```env
OPENAI_API_KEY=your_api_key_here
```

`OpenAILLM.py` and `QwenLLM.py` both try to load `.env` from the current working directory and from the script directory.

## Generate a Dataset

Default mixed-size dataset:

```bash
python MazeGenerator.py --sizes 9 25 --output_dir dataset
```

15x15 / 17x17 dataset:

```bash
python MazeGenerator.py --sizes 15 17 --output_dir dataset --output_suffix _15_17
```

Useful options:

- `--train`, `--val`, `--test`: split sizes
- `--sizes`: odd maze sizes to generate
- `--output_dir`: output directory
- `--output_suffix`: suffix appended to `train/val/test` filenames
- `--sample_size`: print one sample maze before generating the dataset
- `--seed`: random seed

Generated files follow this pattern:

- `dataset/train.json`, `dataset/val.json`, `dataset/test.json`
- `dataset/train_15_17.json`, `dataset/val_15_17.json`, `dataset/test_15_17.json`

## Train

Both training scripts support:

- `--algo`: `A2C` or `REINFORCE`
- `--dataset`: training dataset path
- `--reward_mode`: one of the seven reward combinations
- `--maze_size`: maze size filter within the dataset
- `--episodes`: number of training episodes
- `--lr`: learning rate
- `--entropy_coef`: entropy regularization coefficient
- `--max_steps`: max steps per episode
- `--seed`: random seed
- `--run_name`: tag used in output filenames
- `--top_success_gifs`: number of best successful trajectories to export
- `--progress_every`: progress print interval
- `--heartbeat_seconds`: heartbeat interval during long episodes

### Local Qwen pipeline

`main.py` uses the local `Qwen/Qwen2.5-7B-Instruct` reward backend through `transformers`.

Example:

```bash
python main.py --dataset dataset/train.json --algo A2C --reward_mode sparse_dense_llm --maze_size 9 --episodes 4000 --lr 0.001 --max_steps 500 --seed 42 --run_name qwen_9x9
```

### OpenAI pipeline

`main_openai.py` uses OpenAI for the LLM reward component. Extra options:

- `--llm_model_name`: OpenAI model name, default `gpt-4o-mini`
- `--cache_file`: JSON cache file for LLM reward reuse

Example:

```bash
python main_openai.py --dataset dataset/train.json --algo A2C --reward_mode dense_llm --maze_size 9 --episodes 4000 --lr 0.001 --max_steps 500 --seed 42 --run_name openai_9x9 --llm_model_name gpt-4o-mini --cache_file llm_cache_9.json
```

## Evaluate

Single-maze evaluation:

```bash
python test_agent.py --checkpoint checkpoints/A2C/<checkpoint>.pt --dataset dataset/test.json --maze_size 9 --deterministic --save_gif
```

Wrapper that auto-picks the latest checkpoint if `--checkpoint` is omitted:

```bash
python run_test.py --dataset dataset/test.json --maze_size 9 --deterministic --save_gif
```

Batch evaluation over all mazes of one size:

```bash
python run_test_all.py --checkpoint checkpoints/A2C/<checkpoint>.pt --dataset dataset/test.json --maze_size 9 --deterministic
```

Useful evaluation options:

- `--checkpoint`: path to a saved checkpoint
- `--dataset`: evaluation dataset path
- `--maze_size`: optional size override
- `--max_steps`: optional max-steps override
- `--seed`: evaluation seed override
- `--deterministic`: use greedy action selection
- `--save_gif` or `--save_gifs`: export trajectory GIFs

`run_test_all.py` writes:

- one per-maze detail CSV
- one summary CSV with success rate, average steps, average sparse reward, and average steps on success

## Outputs

Training creates:

- `logs/<algo>/run_<run_tag>.csv`
- `checkpoints/<algo>/<run_tag>.pt`
- `plots/<algo>/learning_curves_<run_tag>.png`
- `gifs/<algo>/...gif`

Evaluation creates:

- `eval_results/<algo>/<timestamp or prefix>_*.csv`

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python MazeGenerator.py --sizes 9 25 --output_dir dataset
python main_openai.py --dataset dataset/train.json --algo A2C --reward_mode dense_llm --maze_size 9 --episodes 4000 --lr 0.001 --max_steps 500 --seed 42 --run_name openai_9x9 --llm_model_name gpt-4o-mini --cache_file llm_cache_9.json
python run_test_all.py --checkpoint checkpoints/A2C/<checkpoint>.pt --dataset dataset/test.json --maze_size 9 --deterministic
```
