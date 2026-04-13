import argparse
import csv
import json
import os
import random

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


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained RL maze agent.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the saved model checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset/test.json",
        help="Path to the maze dataset used for evaluation.",
    )
    parser.add_argument(
        "--maze_size",
        type=int,
        default=None,
        help="Override maze size filter. Defaults to the checkpoint metadata.",
    )
    parser.add_argument(
        "--maze_id",
        type=str,
        default=None,
        help="Optional maze id to evaluate. If omitted, a random maze is chosen from the dataset.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override the maximum steps. Defaults to the checkpoint metadata.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use argmax action selection instead of sampling.",
    )
    parser.add_argument(
        "--save_gif",
        action="store_true",
        help="Save an evaluation trajectory GIF.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible evaluation. Defaults to checkpoint seed.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="eval",
        help="Optional tag to distinguish evaluation outputs.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    algo = checkpoint["algo"]
    maze_size = args.maze_size if args.maze_size is not None else checkpoint["maze_size"]
    max_steps = args.max_steps if args.max_steps is not None else checkpoint["max_steps"]
    seed = args.seed if args.seed is not None else checkpoint.get("seed", 42)
    set_global_seed(seed)

    network, agent = build_agent(algo)
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()

    eval_dir = ensure_method_dirs("eval_results", algo)
    gif_dir = ensure_method_dirs("gifs", algo)

    with open(args.dataset, "r", encoding="utf-8") as f:
        full_dataset = json.load(f)

    maze_dataset = [maze for maze in full_dataset if maze["size"] == maze_size]
    if not maze_dataset:
        raise ValueError(f"No mazes with size {maze_size} were found in {args.dataset}.")

    if args.maze_id is not None:
        matching = [maze for maze in maze_dataset if str(maze["id"]) == str(args.maze_id)]
        if not matching:
            raise ValueError(f"Maze id {args.maze_id} was not found for size {maze_size}.")
        current_maze_data = matching[0]
    else:
        current_maze_data = random.choice(maze_dataset)

    env = MazeEnv(maze_map=current_maze_data["grid"], max_steps=max_steps)
    obs, info = env.reset()
    agent.reset_memory()

    done = False
    total_reward = 0.0
    episode_frames = [env.current_map.copy()]

    while not done:
        with torch.no_grad():
            action = agent.act(obs, info["has_key"], deterministic=args.deterministic)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        frame = env.current_map.copy()
        frame[info["agent_pos"]] = 5
        episode_frames.append(frame)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Algorithm: {algo}")
    print(f"Checkpoint Seed: {checkpoint.get('seed', 'N/A')}")
    print(f"Evaluation Seed: {seed}")
    print(f"Maze ID: {current_maze_data['id']}")
    print(f"Maze Size: {current_maze_data['size']}x{current_maze_data['size']}")
    print(f"Deterministic: {args.deterministic}")
    print(f"Steps: {env.current_step}")
    print(f"Success: {info['is_success']}")
    print(f"Sparse Reward Sum: {total_reward:.2f}")

    run_id = checkpoint_timestamp(args.checkpoint) or "unknown"
    result_filename = os.path.join(
        eval_dir,
        f"{run_id}_maze{current_maze_data['id']}_seed{seed}.csv",
    )
    with open(result_filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Algorithm",
                "Checkpoint",
                "Checkpoint_Seed",
                "Evaluation_Seed",
                "Maze_ID",
                "Maze_Size",
                "Deterministic",
                "Steps",
                "Success",
                "Sparse_Reward_Sum",
            ]
        )
        writer.writerow(
            [
                algo,
                args.checkpoint,
                checkpoint.get("seed", ""),
                seed,
                current_maze_data["id"],
                current_maze_data["size"],
                int(args.deterministic),
                env.current_step,
                int(info["is_success"]),
                round(total_reward, 4),
            ]
        )
    print(f"Saved evaluation summary to {result_filename}")

    if args.save_gif:
        gif_name = (
            f"{args.run_name}_{algo}_maze{current_maze_data['id']}_"
            f"{current_maze_data['size']}x{current_maze_data['size']}_seed{seed}.gif"
        )
        save_trajectory_gif(
            episode_frames,
            current_maze_data["id"],
            filename=gif_name,
            fps=10,
            output_dir=gif_dir,
        )
        print(f"Saved GIF to {os.path.join(gif_dir, gif_name)}")


if __name__ == "__main__":
    main()
