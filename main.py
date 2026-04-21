import argparse
import csv
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from A2C import A2CAgent, A2CNetwork
from Maze import MazeEnv
from REINFORCE import REINFORCEAgent, REINFORCENetwork
from reward_manager import RewardManager
from reward_config import (
    DEFAULT_REWARD_MODE,
    get_reward_mode_choices,
)
from utils import ensure_method_dirs, plot_learning_curves, save_trajectory_gif, set_global_seed


def save_checkpoint(path, network, args, episode):
    checkpoint = {
        "algo": args.algo,
        "reward_mode": args.reward_mode,
        "maze_size": int(args.maze_size),
        "episodes_trained": episode,
        "lr": args.lr,
        "entropy_coef": args.entropy_coef,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "run_name": args.run_name,
        "model_state_dict": network.state_dict(),
    }
    torch.save(checkpoint, path)


if __name__ == "__main__":
    # 1. Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Train an RL agent for maze navigation.")
    parser.add_argument(
        "--algo",
        type=str,
        default="A2C",
        choices=["A2C", "REINFORCE"],
        help="Training algorithm to use.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset/train.json",
        help="Path to the maze dataset used for training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate.",
    )
    parser.add_argument(
        "--entropy_coef",
        type=float,
        default=0.05,
        help="Entropy regularization coefficient.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=500,
        help="Maximum number of steps allowed per episode.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=2000,
        help="Number of training episodes.",
    )
    parser.add_argument(
        "--maze_size",
        type=int,
        default=9,
        help="Maze size to filter from the dataset, e.g. 9, 15, 17, 25.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible training runs.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="default",
        help="Optional tag to distinguish experiment outputs.",
    )
    parser.add_argument(
        "--top_success_gifs",
        type=int,
        default=3,
        help="Number of best successful trajectories to save as GIFs.",
    )
    parser.add_argument(
        "--reward_mode",
        type=str,
        default=DEFAULT_REWARD_MODE,
        choices=get_reward_mode_choices(),
        help="Reward combination to use during training.",
    )
    args = parser.parse_args()
    set_global_seed(args.seed)

    # 2. Prepare the CSV log file.
    logs_dir = ensure_method_dirs("logs", args.algo)
    checkpoints_dir = ensure_method_dirs("checkpoints", args.algo)
    plots_dir = ensure_method_dirs("plots", args.algo)
    gifs_dir = ensure_method_dirs("gifs", args.algo)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = (
        f"{args.run_name}_{args.algo}_{args.reward_mode}_size{args.maze_size}_"
        f"lr{args.lr}_seed{args.seed}_{run_timestamp}"
    )
    log_filename = os.path.join(logs_dir, f"run_{run_tag}.csv")
    checkpoint_filename = os.path.join(checkpoints_dir, f"{run_tag}.pt")
    plot_filename = f"learning_curves_{run_tag}.png"
    plot_filepath = os.path.join(plots_dir, plot_filename)

    with open(log_filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Episode",
                "Maze_ID",
                "Maze_Size",
                "Steps",
                "Total_Reward",
                "Sparse_Reward",
                "Dense_Reward",
                "LLM_Reward",
                "LLM_Range_Min",
                "LLM_Range_Max",
                "Success",
                "Loss",
                "Algo",
                "Seed",
                "Run_Name",
                "Reward_Mode",
            ]
        )

    print(f"Training log will be written to: {log_filename}")
    print(f"Checkpoint will be written to: {checkpoint_filename}")
    print(f"Run seed: {args.seed}")
    print(f"Reward mode: {args.reward_mode}")

    # 3. Load the maze dataset.
    if not os.path.exists(args.dataset):
            raise FileNotFoundError(f"can not find the dataset {args.dataset}！run MazeGenerator.py first")
    with open(args.dataset, 'r', encoding='utf-8') as f:
        full_dataset = json.load(f)
    target_size = int(args.maze_size)
    maze_dataset = [m for m in full_dataset if m["size"] == target_size]
    if not maze_dataset:
        raise ValueError(f"No mazes with size {target_size} were found in {args.dataset}.")
 
    print(f"load {args.maze_size}x{args.maze_size} numbers of dataset {len(maze_dataset)} for training")

    # 4. Initialize the model, optimizer, and reward modules.
    if args.algo == "A2C":
        network = A2CNetwork()
        agent = A2CAgent(network)
    else:
        network = REINFORCENetwork()
        agent = REINFORCEAgent(network)

    optimizer = optim.Adam(network.parameters(), lr=args.lr)
    reward_manager = RewardManager(args.reward_mode, llm_model_name="Qwen/Qwen2.5-7B-Instruct")

    gamma = 0.99
    num_episodes = args.episodes

    history_rewards = []
    history_steps = []
    history_keys = []
    history_successes = []

    print(
        f"Starting {args.algo} training for {num_episodes} episodes "
        f"(lr={args.lr}, max_steps={args.max_steps}, seed={args.seed}).\n"
    )
    successful_episodes = []

    for episode in range(1, num_episodes + 1):
        # Sample a maze instance for the current episode.
        current_maze_data = random.choice(maze_dataset)
        maze_id = current_maze_data["id"]
        maze_size = current_maze_data["size"]
        maze_grid = current_maze_data["grid"]

        # Build a fresh environment from the sampled maze.
        env = MazeEnv(maze_map=maze_grid, max_steps=args.max_steps)

        obs, prev_info = env.reset()
        agent.reset_memory()
        reward_manager.reset()
        reward_manager.reward_engine.initialize_episode(maze_grid, prev_info["key_pos"], prev_info["exit_pos"])

        log_probs, values, rewards, entropies = [], [], [], []
        episode_frames = [env.current_map.copy()]
        episode_sparse_total = 0.0
        episode_dense_total = 0.0
        episode_llm_total = 0.0
        episode_llm_min_total = 0.0
        episode_llm_max_total = 0.0

        done = False
        while not done:
            if args.algo == "A2C":
                action, log_prob, value, entropy = agent.select_action(obs, prev_info["has_key"])
                values.append(value)
            else:
                action, log_prob, entropy = agent.select_action(obs, prev_info["has_key"])

            next_obs, sparse_reward, terminated, truncated, current_info = env.step(action)
            done = terminated or truncated

            frame = env.current_map.copy()
            frame[current_info["agent_pos"]] = 5
            episode_frames.append(frame)

            total_step_reward, reward_components, llm_reward_range = reward_manager.compute_step_reward(
                current_info, prev_info
            )

            log_probs.append(log_prob)
            rewards.append(total_step_reward)
            entropies.append(entropy)
            episode_sparse_total += reward_components["sparse"]
            episode_dense_total += reward_components["dense"]
            episode_llm_total += reward_components["llm"]
            episode_llm_min_total += llm_reward_range["min"]
            episode_llm_max_total += llm_reward_range["max"]

            obs = next_obs
            prev_info = current_info

        # Compute discounted returns.
        discounted_return = 0
        returns = []
        for reward in rewards[::-1]:
            discounted_return = reward + gamma * discounted_return
            returns.insert(0, discounted_return)
        returns = torch.tensor(returns)

        log_probs = torch.cat(log_probs)
        entropies = torch.cat(entropies)

        # Compute the training loss.
        if args.algo == "A2C":
            values = torch.cat(values).squeeze()
            advantages = returns - values.detach()
            actor_loss = -(log_probs * advantages).mean()
            critic_loss = F.mse_loss(values, returns)
            entropy_loss = entropies.mean()
            loss = actor_loss + 0.5 * critic_loss - args.entropy_coef * entropy_loss
        else:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            policy_loss = -(log_probs * returns).mean()
            entropy_loss = entropies.mean()
            loss = policy_loss - args.entropy_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record episode statistics.
        episode_key_acquired = 1.0 if current_info.get("has_key", False) else 0.0
        episode_reward = sum(rewards)
        episode_steps = env.current_step
        episode_success = 1.0 if current_info["is_success"] else 0.0

        history_rewards.append(episode_reward)
        history_steps.append(episode_steps)
        history_keys.append(episode_key_acquired)
        history_successes.append(episode_success)

        with open(log_filename, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    episode,
                    maze_id,
                    maze_size,
                    episode_steps,
                    round(episode_reward, 4),
                    round(episode_sparse_total, 4),
                    round(episode_dense_total, 4),
                    round(episode_llm_total, 4),
                    round(episode_llm_min_total, 4),
                    round(episode_llm_max_total, 4),
                    int(episode_success),
                    round(loss.item(), 4),
                    args.algo,
                    args.seed,
                    args.run_name,
                    args.reward_mode,
                ]
            )

        # Record successful trajectories and save only the best ones after training.
        if current_info["is_success"]:
            successful_episodes.append(
                {
                    "episode": episode,
                    "maze_size": maze_size,
                    "reward": episode_reward,
                    "steps": episode_steps,
                    "frames": [frame.copy() for frame in episode_frames],
                }
            )

        if episode % 500 == 0:
            save_trajectory_gif(
                episode_frames,
                episode,
                filename=f"traj_{run_tag}_{maze_size}x{maze_size}_ep{episode}.gif",
                fps=15,
                output_dir=gifs_dir,
            )

        if episode % 100 == 0:
            avg_reward = np.mean(history_rewards[-100:])
            avg_keys = np.mean(history_keys[-100:])
            avg_success = np.mean(history_successes[-100:])
            print(
                f"Episode {episode:4d} | Avg Reward: {avg_reward:.2f} | "
                f"Key Rate: {avg_keys * 100:.1f}% | "
                f"Success Rate: {avg_success * 100:.1f}% | Loss: {loss.item():.4f} | "
                f"Current Maze Size: {maze_size}x{maze_size}"
            )

    if successful_episodes:
        successful_episodes.sort(
            key=lambda item: (-item["reward"], item["steps"], item["episode"])
        )
        best_successes = successful_episodes[: args.top_success_gifs]
        print(f"\nSaving top {len(best_successes)} successful trajectory GIFs...")
        for rank, result in enumerate(best_successes, start=1):
            save_trajectory_gif(
                result["frames"],
                result["episode"],
                filename=(
                    f"best{rank}_{run_tag}_SUCCESS_{result['maze_size']}x"
                    f"{result['maze_size']}_ep{result['episode']}_"
                    f"reward{result['reward']:.2f}_steps{result['steps']}.gif"
                ),
                fps=20,
                output_dir=gifs_dir,
            )
    else:
        print("\nNo successful episodes were found, so no success GIFs were saved.")

    plot_learning_curves(
        history_rewards,
        history_steps,
        history_successes,
        plot_filename,
        output_dir=plots_dir,
    )
    save_checkpoint(checkpoint_filename, network, args, num_episodes)
    print(
        f"\nTraining completed. Learning curves were generated "
        f"(output path: {plot_filepath})."
    )
    print(f"Model checkpoint saved to: {checkpoint_filename}")
