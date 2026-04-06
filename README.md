# MazeAgentRL

MazeAgentRL is a reinforcement learning project for maze navigation under partial observability.  
The agent must:

1. find the key
2. pick up the key
3. reach the exit

The project currently supports:

- `A2C`
- `REINFORCE`
- sparse reward
- hand-crafted dense reward
- LLM-based reward shaping

The repository also includes:

- dataset generation
- dataset inspection
- training logs
- learning-curve plots
- trajectory GIF generation
- single-maze evaluation
- full test-set batch evaluation

## 1. Project Structure

Main files:

- `main.py`: training entry point
- `test_agent.py`: evaluate one maze with a trained checkpoint
- `run_test.py`: convenient wrapper for single evaluation on `dataset/test.json`
- `run_test_all.py`: batch evaluation on the full `dataset/test.json`
- `Maze.py`: maze environment
- `A2C.py`: A2C model and agent
- `REINFORCE.py`: REINFORCE model and agent
- `RewardEngine.py`: dense reward and bounded LLM reward sampling
- `reward_manager.py`: combines sparse, dense, and LLM rewards
- `reward_config.py`: reward modes and reward constants
- `OpenAILLM.py`: OpenAI-based reward range generator with cache
- `MazeGenerator.py`: generate train/val/test maze datasets
- `inspect_dataset.py`: inspect mazes visually
- `utils.py`: plotting, seeding, GIF saving
- `BFS_solver.py`: BFS baseline for shortest-path style analysis

Common output folders:

- `dataset/`: generated maze datasets
- `logs/`: training CSV logs
- `checkpoints/`: saved model checkpoints
- `plots/`: learning-curve figures
- `gifs/`: training and evaluation GIFs
- `eval_results/`: evaluation CSV outputs

## 2. Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

If you use LLM reward during training, set your OpenAI API key first.

Linux / macOS:

```bash
export OPENAI_API_KEY=your_api_key
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

## 3. Dataset Generation and Inspection

Generate train / validation / test datasets:

```bash
python MazeGenerator.py
```

Inspect sampled mazes:

```bash
python inspect_dataset.py
```

Generated dataset files:

- `dataset/train.json`
- `dataset/val.json`
- `dataset/test.json`

Each maze record contains at least:

- maze id
- maze size
- maze grid

## 4. Problem Setting

The maze is partially observable.  
The agent only sees a local `3 x 3` observation centered on itself.

Tile encoding:

- `0`: path
- `1`: wall
- `2`: start
- `3`: exit
- `4`: key
- `5`: agent marker in observation / GIF rendering

Action encoding:

- `0`: up
- `1`: down
- `2`: left
- `3`: right

Episode termination:

- success: the agent reaches the exit after obtaining the key
- truncation: the agent hits `max_steps`

## 5. Reward Design

The training reward can combine three sources:

```text
total_step_reward = sparse_reward + dense_reward + llm_reward
```

### Sparse reward

Environment milestone reward:

- reward when the key is picked up
- reward when the exit is reached after collecting the key

### Dense reward

Defined in `RewardEngine.py`. It includes:

- step penalty
- revisit penalty
- progress bonus based on weighted distance to the current target

Current target rule:

- before key pickup: target is the key
- after key pickup: target is the exit

### LLM reward

Defined through `OpenAILLM.py` and `RewardEngine.py`.

The LLM:

- observes the full maze state as text
- compares previous position and current position
- estimates whether the agent moved closer to the current goal
- returns a bounded reward range

Then the reward engine:

- clips the range
- samples one scalar value from the range
- enforces an episode-level LLM reward budget

## 6. Training

Training is done with `main.py`.

### Basic training command

Train `A2C` on `9x9` mazes:

```bash
python main.py --algo A2C --maze_size 9 --dataset dataset/train.json --lr 0.001 --entropy_coef 0.05 --max_steps 500 --episodes 1500
```

Train `REINFORCE` on `9x9` mazes:

```bash
python main.py --algo REINFORCE --maze_size 9 --dataset dataset/train.json --lr 0.001 --entropy_coef 0.05 --max_steps 500 --episodes 1500
```

Example with explicit run name and reward mode:

```bash
python main.py --algo REINFORCE --maze_size 9 --dataset dataset/train.json --lr 0.001 --entropy_coef 0.05 --max_steps 500 --episodes 1500 --reward_mode sparse_dense_llm --run_name exp_reward_compare
```

### Training options

`main.py` supports these main arguments:

- `--algo`: training algorithm, choices: `A2C`, `REINFORCE`
- `--dataset`: training dataset path, default: `dataset/train.json`
- `--lr`: learning rate
- `--entropy_coef`: entropy regularization coefficient
- `--max_steps`: maximum steps allowed per episode
- `--episodes`: number of training episodes
- `--maze_size`: maze size filter, choices: `9`, `25`
- `--seed`: random seed
- `--run_name`: custom tag for output files
- `--top_success_gifs`: number of best successful training trajectories to save as GIFs
- `--reward_mode`: reward combination mode, choices come from `reward_config.py`

### Reward mode

The exact choices are defined in `reward_config.py`.  
Typical usage is to compare different reward combinations such as:

- sparse only
- sparse + dense
- sparse + dense + llm

You should check `reward_config.py` if you want the full current list of allowed mode names.

### What training produces

Training will generate:

- CSV logs in `logs/<algo>/`
- checkpoint `.pt` files in `checkpoints/<algo>/`
- learning curves in `plots/<algo>/`
- trajectory GIFs in `gifs/<algo>/`

The training CSV includes fields such as:

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

## 7. Testing One Maze

If you already have a trained checkpoint and want to test it on one maze, use `test_agent.py`.

### Basic single-maze test

```bash
python test_agent.py --checkpoint checkpoints/REINFORCE/your_model.pt --dataset dataset/test.json --deterministic --save_gif
```

### Common single-maze test examples

Test a specific maze id:

```bash
python test_agent.py --checkpoint checkpoints/REINFORCE/your_model.pt --dataset dataset/test.json --maze_id 81 --deterministic --save_gif
```

Override maze size:

```bash
python test_agent.py --checkpoint checkpoints/REINFORCE/your_model.pt --dataset dataset/test.json --maze_size 9 --deterministic
```

Override evaluation seed:

```bash
python test_agent.py --checkpoint checkpoints/REINFORCE/your_model.pt --dataset dataset/test.json --seed 42 --deterministic
```

### Single-maze test options

- `--checkpoint`: path to a trained `.pt` checkpoint, required
- `--dataset`: evaluation dataset path, default: `dataset/test.json`
- `--maze_size`: override maze size; if omitted, use checkpoint metadata
- `--maze_id`: evaluate one specific maze id
- `--max_steps`: override max steps; if omitted, use checkpoint metadata
- `--deterministic`: use argmax action selection instead of sampling
- `--save_gif`: save the trajectory as a GIF
- `--seed`: override evaluation seed
- `--run_name`: tag for evaluation output file names

### Single-maze outputs

`test_agent.py` prints:

- checkpoint path
- algorithm
- checkpoint seed
- evaluation seed
- maze id
- maze size
- whether deterministic mode is used
- steps
- success / failure
- sparse reward sum

It also saves:

- one CSV under `eval_results/<algo>/`
- one GIF under `gifs/<algo>/` if `--save_gif` is enabled

## 8. Convenient Single Test Wrapper

If you do not want to manually type the checkpoint path every time, use `run_test.py`.

It automatically:

- defaults to `dataset/test.json`
- finds the latest checkpoint under `checkpoints/` if you do not provide one

### Example

```bash
python run_test.py --deterministic --save_gif
```

Specify a particular checkpoint:

```bash
python run_test.py --checkpoint "checkpoints/REINFORCE/your_model.pt" --maze_id 81 --deterministic --save_gif
```

### `run_test.py` options

- `--checkpoint`: optional checkpoint path
- `--dataset`: evaluation dataset path, default: `dataset/test.json`
- `--maze_id`: optional maze id
- `--maze_size`: optional maze size override
- `--max_steps`: optional max-step override
- `--seed`: optional evaluation seed override
- `--run_name`: output tag
- `--deterministic`: use greedy action selection
- `--save_gif`: save one evaluation GIF

## 9. Batch Testing the Full Test Set

If you want to evaluate your trained model on all mazes in `dataset/test.json`, use `run_test_all.py`.

### Basic batch evaluation

```bash
python run_test_all.py --deterministic
```

### Batch evaluation with GIFs

```bash
python run_test_all.py --deterministic --save_gifs
```

### Batch evaluation with explicit checkpoint

```bash
python run_test_all.py --checkpoint "checkpoints/REINFORCE/your_model.pt" --maze_size 9 --deterministic
```

### `run_test_all.py` options

- `--checkpoint`: optional checkpoint path
- `--dataset`: evaluation dataset path, default: `dataset/test.json`
- `--maze_size`: optional maze size filter; default comes from checkpoint
- `--max_steps`: optional max-step override
- `--seed`: optional evaluation seed override
- `--run_name`: output tag
- `--deterministic`: use greedy action selection
- `--save_gifs`: save one GIF per maze

### Batch evaluation outputs

`run_test_all.py` saves:

- one detailed CSV for every maze result
- one summary CSV with aggregate metrics
- optional GIFs for all evaluated mazes

Typical summary metrics:

- number of mazes
- success rate
- average steps
- average sparse reward
- average steps on successful episodes

## 10. GIF Visualization

Trajectory GIFs are generated by `utils.py`.

Current GIF behavior:

- shows the maze trajectory frame by frame
- shows the current step index
- shows the total number of steps
- shows whether the episode is still `Running` or already `Finished`

This makes it easier to see:

- how far the agent has progressed
- whether the agent solved the maze
- whether the GIF ended because the trajectory finished

## 11. Model Details

### Shared design

Both `A2C` and `REINFORCE` use:

- symbolic observation embedding
- flattened `3 x 3` local view
- `GRU` recurrent memory

This design is important because the environment is partially observable and the agent cannot see the full maze at once.

### A2C

`A2C` includes:

- actor head
- critic head
- entropy regularization

It generally provides lower-variance updates because it uses a value function baseline.

### REINFORCE

`REINFORCE` includes:

- policy head only
- normalized discounted returns
- entropy regularization

It is simpler and serves as a useful baseline for comparison.

## 12. Other Functions in This Project

Besides training and testing, the project also provides several useful utilities.

### `MazeGenerator.py`

Generates random mazes for:

- training
- validation
- testing

Useful when:

- you want more mazes
- you want to regenerate a new dataset split
- you want to compare different random map sets

### `inspect_dataset.py`

Visual inspection tool for generated mazes.

Useful when:

- you want to check whether start / key / exit placement is reasonable
- you want to inspect maze difficulty
- you want to verify that generation worked correctly

### `BFS_solver.py`

Provides a BFS baseline.

Useful when:

- you want an approximate classical baseline
- you want to compare RL behavior against shortest-path style planning
- you want to analyze whether the learned agent is close to optimal on simple mazes

### `llm_cache.json`

Stores LLM reward results for repeated transitions.

Useful when:

- you want to reduce repeated API calls
- you want to reduce cost
- you want more stable reward reuse across repeated transitions

## 13. Typical Workflow

Recommended end-to-end workflow:

1. install dependencies
2. set `OPENAI_API_KEY` if using LLM reward
3. generate dataset with `MazeGenerator.py`
4. inspect mazes with `inspect_dataset.py`
5. train with `main.py`
6. test one maze with `test_agent.py` or `run_test.py`
7. test the full test set with `run_test_all.py`
8. analyze CSV logs, plots, and GIFs

## 14. Notes and Limitations

- `test_agent.py` and the test wrappers evaluate environment reward behavior, mainly sparse task completion behavior.
- To load a checkpoint successfully, the model architecture must match the checkpoint.
- If you modify the network architecture in `A2C.py` or `REINFORCE.py`, old checkpoints may no longer load.
- LLM reward is used during training, not for direct action generation during testing.
- If you use old cached LLM reward values, `llm_cache.json` format may affect whether logged LLM min/max look like a true range or a point value.

## 15. Summary

This project is a complete RL training and evaluation pipeline for partially observable maze navigation.

It supports:

- dataset generation
- reward-ablation style training
- checkpoint saving
- single-maze evaluation
- full test-set evaluation
- trajectory GIF visualization
- plot and CSV based analysis

If your goal is just to start quickly, use these three commands:

Train:

```bash
python main.py --algo REINFORCE --maze_size 9 --dataset dataset/train.json --lr 0.001 --entropy_coef 0.05 --max_steps 500 --episodes 1500 --reward_mode sparse_dense_llm --run_name exp1
```

Test one maze:

```bash
python run_test.py --deterministic --save_gif
```

Test the whole test set:

```bash
python run_test_all.py --deterministic
```
