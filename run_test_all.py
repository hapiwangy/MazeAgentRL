import argparse
import csv
import glob
import json
import os
import statistics

import torch

from A2C import A2CAgent, A2CNetwork
from Maze import MazeEnv
from REINFORCE import REINFORCEAgent, REINFORCENetwork
from utils import checkpoint_timestamp, ensure_method_dirs, save_trajectory_gif, set_global_seed


def build_agent(algo):
    if algo == "A2C":
        network = A2CNetwork()
        agent = A2CAgent(network)
    elif algo == "REINFORCE":
        network = REINFORCENetwork()
        agent = REINFORCEAgent(network)
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")
    return network, agent


def find_latest_checkpoint(checkpoint_dir):
    checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, "*", "*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoint files were found under {checkpoint_dir}.")
    return max(checkpoint_paths, key=os.path.getmtime)


def evaluate_maze(agent, maze_data, max_steps, deterministic):
    env = MazeEnv(maze_map=maze_data["grid"], max_steps=max_steps)
    obs, info = env.reset()
    agent.reset_memory()

    done = False
    total_reward = 0.0
    frames = [env.current_map.copy()]
    final_info = info

    while not done:
        with torch.no_grad():
            action = agent.act(obs, info["has_key"], deterministic=deterministic)

        obs, reward, terminated, truncated, final_info = env.step(action)
        info = final_info
        done = terminated or truncated
        total_reward += reward

        frame = env.current_map.copy()
        frame[final_info["agent_pos"]] = 5
        frames.append(frame)

    return {
        "maze_id": maze_data["id"],
        "maze_size": maze_data["size"],
        "steps": env.current_step,
        "success": int(final_info["is_success"]),
        "sparse_reward_sum": float(total_reward),
        "frames": frames,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint on every maze in dataset/test.json."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path. If omitted, the latest checkpoint under checkpoints/ is used.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset/test.json",
        help="Evaluation dataset path. Defaults to dataset/test.json.",
    )
    parser.add_argument(
        "--maze_size",
        type=int,
        default=None,
        help="Optional maze size filter. Defaults to checkpoint metadata.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional max-steps override. Defaults to checkpoint metadata.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional evaluation seed override. Defaults to checkpoint seed.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="eval_test_all",
        help="Tag for evaluation outputs.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use greedy argmax action selection during evaluation.",
    )
    parser.add_argument(
        "--save_gifs",
        action="store_true",
        help="Save one GIF per maze evaluation.",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or find_latest_checkpoint("checkpoints")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    algo = checkpoint["algo"]
    maze_size = args.maze_size if args.maze_size is not None else checkpoint["maze_size"]
    max_steps = args.max_steps if args.max_steps is not None else checkpoint["max_steps"]
    seed = args.seed if args.seed is not None else checkpoint.get("seed", 42)
    set_global_seed(seed)

    network, agent = build_agent(algo)
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()

    with open(args.dataset, "r", encoding="utf-8") as f:
        full_dataset = json.load(f)

    maze_dataset = [maze for maze in full_dataset if maze["size"] == maze_size]
    if not maze_dataset:
        raise ValueError(f"No mazes with size {maze_size} were found in {args.dataset}.")

    eval_dir = ensure_method_dirs("eval_results", algo)
    gif_dir = ensure_method_dirs("gifs", algo)

    run_id = checkpoint_timestamp(checkpoint_path) or "unknown"
    detail_path = os.path.join(eval_dir, f"{run_id}_seed{seed}_details.csv")
    summary_path = os.path.join(eval_dir, f"{run_id}_seed{seed}_summary.csv")

    results = []
    for maze_data in maze_dataset:
        result = evaluate_maze(agent, maze_data, max_steps, args.deterministic)
        results.append(result)

        if args.save_gifs:
            gif_name = (
                f"{args.run_name}_{algo}_maze{result['maze_id']}_"
                f"{result['maze_size']}x{result['maze_size']}_seed{seed}.gif"
            )
            save_trajectory_gif(
                result["frames"],
                result["maze_id"],
                filename=gif_name,
                fps=10,
                output_dir=gif_dir,
            )

    with open(detail_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Algorithm",
                "Checkpoint",
                "Evaluation_Seed",
                "Deterministic",
                "Maze_ID",
                "Maze_Size",
                "Steps",
                "Success",
                "Sparse_Reward_Sum",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    algo,
                    checkpoint_path,
                    seed,
                    int(args.deterministic),
                    result["maze_id"],
                    result["maze_size"],
                    result["steps"],
                    result["success"],
                    round(result["sparse_reward_sum"], 4),
                ]
            )

    success_rate = sum(item["success"] for item in results) / len(results)
    avg_steps = statistics.mean(item["steps"] for item in results)
    avg_reward = statistics.mean(item["sparse_reward_sum"] for item in results)
    successful_steps = [item["steps"] for item in results if item["success"]]
    avg_success_steps = statistics.mean(successful_steps) if successful_steps else ""

    with open(summary_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Algorithm",
                "Checkpoint",
                "Maze_Size",
                "Num_Mazes",
                "Evaluation_Seed",
                "Deterministic",
                "Success_Rate",
                "Average_Steps",
                "Average_Sparse_Reward",
                "Average_Steps_On_Success",
            ]
        )
        writer.writerow(
            [
                algo,
                checkpoint_path,
                maze_size,
                len(results),
                seed,
                int(args.deterministic),
                round(success_rate, 4),
                round(avg_steps, 4),
                round(avg_reward, 4),
                round(avg_success_steps, 4) if avg_success_steps != "" else "",
            ]
        )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Algorithm: {algo}")
    print(f"Maze size: {maze_size}")
    print(f"Num mazes: {len(results)}")
    print(f"Deterministic: {args.deterministic}")
    print(f"Success rate: {success_rate * 100:.2f}%")
    print(f"Average steps: {avg_steps:.2f}")
    print(f"Average sparse reward: {avg_reward:.2f}")
    if avg_success_steps != "":
        print(f"Average steps on success: {avg_success_steps:.2f}")
    else:
        print("Average steps on success: N/A")
    print(f"Saved details to {detail_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
