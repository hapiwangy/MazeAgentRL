## Optimized Bundle

This folder is a standalone accelerated copy of the training pipeline.

It preserves the original experiment logic:
- same maze task
- same A2C / REINFORCE design
- same reward modes
- same sparse / dense / LLM reward semantics
- same dataset format

It changes runtime behavior only to remove avoidable overhead:
- use GPU tensors automatically when available
- skip text-map generation unless the selected reward mode uses LLM
- reuse BFS distance maps across repeated mazes
- reuse RewardEngine distance features inside LLM reward code instead of recomputing BFS
- reduce per-episode log I/O overhead
- avoid storing full frame copies every step during training and evaluation
- write LLM cache files to this folder consistently
- reduce console spam by default while keeping optional verbose logging

Recommended entry points:
- `python main.py --algo A2C --reward_mode sparse_dense_llm --maze_size 9`
- `python run_test.py --checkpoint checkpoints\\A2C\\<checkpoint>.pt --deterministic`
- `python run_test_all.py --checkpoint checkpoints\\A2C\\<checkpoint>.pt --deterministic`

Optional logging controls:
- `set QWEN_LLM_VERBOSE=1` to print every Qwen LLM call and response
- `set OPENAI_LLM_VERBOSE=1` to print every OpenAI LLM call and response

Sequential checklist runner:

- `python run_pending_experiments.py --script script.txt`
- It reads `script.txt`, skips lines already marked as `[V]`, runs `[ ]` lines one by one, and marks each successful command as `[V]`.
- `python run_pending_experiments.py --script script.txt --dry_run` shows what would run without starting jobs.
- The runner now separates `llm` and non-`llm` jobs: by default it runs up to 4 non-LLM experiments in parallel and 1 LLM experiment at a time.
- You can tune this with `--max_parallel_non_llm` and `--max_parallel_llm`.

## CARC batch jobs

This bundle now includes Slurm scripts under `carc/`:

- `carc/train_maze.sbatch`: submit one training run
- `carc/eval_maze.sbatch`: evaluate one checkpoint on the test split

Example training submission:

```bash
cd ~/MazeAgentRL/optimized_bundle
sbatch --export=ALL,PROJECT_DIR=$PWD,ALGO=A2C,REWARD_MODE=sparse,MAZE_SIZE=9,EPISODES=4000,LR=0.001,MAX_STEPS=500,SEED=42,RUN_NAME=carc_a2c_sparse carc/train_maze.sbatch
```

Example evaluation submission:

```bash
cd ~/MazeAgentRL/optimized_bundle
sbatch --export=ALL,PROJECT_DIR=$PWD,CHECKPOINT=$PWD/checkpoints/A2C/<checkpoint>.pt,RUN_NAME=carc_eval,DETERMINISTIC=1 carc/eval_maze.sbatch
```

Important notes for CARC:

- The scripts default to the `gpu` partition and request one GPU because Qwen reward mode uses local transformers inference.
- For non-LLM reward modes you can remove `--gres=gpu:1` and switch to a CPU partition if your CARC account uses a different queue policy.
- `requirements.txt` now includes `transformers` and `accelerate` for `QwenLLM.py`.
- Hugging Face model cache is written under `.hf_cache/` inside `optimized_bundle` unless you override `HF_HOME`.
- If your cluster requires a specific Python or CUDA module, add the corresponding `module load ...` lines near the top of the sbatch script.
