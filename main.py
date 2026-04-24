import argparse
import csv
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from A2C import A2CAgent, A2CNetwork
from Maze import MazeEnv
from REINFORCE import REINFORCEAgent, REINFORCENetwork
from reward_config import DEFAULT_REWARD_MODE, get_reward_mode_choices
from reward_manager import RewardManager
from utils import (
    ensure_method_dirs,
    plot_learning_curves,
    reconstruct_episode_frames,
    save_trajectory_gif,
    set_global_seed,
)


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


def build_agent_and_network(algo, device):
    if algo == "A2C":
        network = A2CNetwork().to(device)
        agent = A2CAgent(network, device=device)
    else:
        network = REINFORCENetwork().to(device)
        agent = REINFORCEAgent(network, device=device)
    return network, agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL agent for maze navigation.")
    parser.add_argument("--algo", type=str, default="A2C", choices=["A2C", "REINFORCE"], help="Training algorithm to use.")
    parser.add_argument("--dataset", type=str, default="dataset/train.json", help="Path to the maze dataset used for training.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--entropy_coef", type=float, default=0.05, help="Entropy regularization coefficient.")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum number of steps allowed per episode.")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of training episodes.")
    parser.add_argument("--maze_size", type=int, default=9, help="Maze size to filter from the dataset, e.g. 9, 15, 17, 25.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible training runs.")
    parser.add_argument("--run_name", type=str, default="default", help="Optional tag to distinguish experiment outputs.")
    parser.add_argument("--top_success_gifs", type=int, default=3, help="Number of best successful trajectories to save as GIFs.")
    parser.add_argument(
        "--progress_every",
        type=int,
        default=10,
        help="Print lightweight training progress every N episodes.",
    )
    parser.add_argument(
        "--heartbeat_seconds",
        type=float,
        default=30.0,
        help="Print an in-episode heartbeat every N seconds so long LLM episodes still show progress.",
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    print(f"Training log will be written to: {log_filename}")
    print(f"Checkpoint will be written to: {checkpoint_filename}")
    print(f"Run seed: {args.seed}")
    print(f"Reward mode: {args.reward_mode}")
    print(f"Device: {device}")

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Cannot find dataset: {args.dataset}. Run MazeGenerator.py first.")

    with open(args.dataset, "r", encoding="utf-8") as f:
        full_dataset = json.load(f)

    target_size = int(args.maze_size)
    maze_dataset = [m for m in full_dataset if m["size"] == target_size]
    if not maze_dataset:
        raise ValueError(f"No mazes with size {target_size} were found in {args.dataset}.")

    print(f"load {args.maze_size}x{args.maze_size} numbers of dataset {len(maze_dataset)} for training")

    network, agent = build_agent_and_network(args.algo, device)
    optimizer = optim.Adam(network.parameters(), lr=args.lr)
    reward_manager = RewardManager(args.reward_mode, llm_model_name="Qwen/Qwen2.5-7B-Instruct")

    gamma = 0.99
    num_episodes = args.episodes
    history_rewards = []
    history_steps = []
    history_keys = []
    history_successes = []
    successful_episodes = []

    print(
        f"Starting {args.algo} training for {num_episodes} episodes "
        f"(lr={args.lr}, max_steps={args.max_steps}, seed={args.seed}).\n"
    )
    train_start_time = time.perf_counter()

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

        for episode in range(1, num_episodes + 1):
            current_maze_data = random.choice(maze_dataset)
            maze_id = current_maze_data["id"]
            maze_size = current_maze_data["size"]
            maze_grid = current_maze_data["grid"]
            episode_start_time = time.perf_counter()
            next_heartbeat_time = episode_start_time + max(1.0, float(args.heartbeat_seconds))

            env = MazeEnv(
                maze_map=maze_grid,
                max_steps=args.max_steps,
                include_text_info=reward_manager.uses_llm,
            )

            obs, prev_info = env.reset()
            reward_manager.initialize_episode(maze_grid, prev_info["key_pos"], prev_info["exit_pos"])
            prev_info = reward_manager.enrich_info(prev_info)
            agent.reset_memory()
            reward_manager.reset()

            log_probs, values, rewards, entropies = [], [], [], []
            track_positions = (episode % 500 == 0) or (args.top_success_gifs > 0)
            episode_positions = [prev_info["agent_pos"]] if track_positions else None
            key_pick_frame_idx = None
            episode_sparse_total = 0.0
            episode_dense_total = 0.0
            episode_llm_total = 0.0
            episode_llm_min_total = 0.0
            episode_llm_max_total = 0.0

            done = False
            current_info = prev_info
            while not done:
                if args.algo == "A2C":
                    action, log_prob, value, entropy = agent.select_action(obs, prev_info["has_key"])
                    values.append(value)
                else:
                    action, log_prob, entropy = agent.select_action(obs, prev_info["has_key"])

                next_obs, _, terminated, truncated, current_info = env.step(action)
                current_info = reward_manager.enrich_info(current_info)
                done = terminated or truncated

                if episode_positions is not None:
                    episode_positions.append(current_info["agent_pos"])
                    if current_info.get("picked_key_this_step") and key_pick_frame_idx is None:
                        key_pick_frame_idx = len(episode_positions) - 1

                total_step_reward, reward_components, llm_reward_range = reward_manager.compute_step_reward(
                    current_info,
                    prev_info,
                )

                now = time.perf_counter()
                if now >= next_heartbeat_time:
                    elapsed_episode = now - episode_start_time
                    print(
                        f"[Heartbeat] ep={episode}/{num_episodes} | step={env.current_step}/{args.max_steps} | "
                        f"maze={maze_id} | elapsed_ep={elapsed_episode / 60:.1f}m | "
                        f"has_key={int(current_info.get('has_key', False))}"
                    )
                    next_heartbeat_time = now + max(1.0, float(args.heartbeat_seconds))

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

            returns = torch.empty(len(rewards), dtype=torch.float32, device=device)
            discounted_return = 0.0
            for idx in range(len(rewards) - 1, -1, -1):
                discounted_return = rewards[idx] + gamma * discounted_return
                returns[idx] = discounted_return

            log_probs = torch.cat(log_probs)
            entropies = torch.cat(entropies)

            if args.algo == "A2C":
                values = torch.cat(values).squeeze(-1)
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

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            episode_key_acquired = 1.0 if current_info.get("has_key", False) else 0.0
            episode_reward = float(sum(rewards))
            episode_steps = env.current_step
            episode_success = 1.0 if current_info["is_success"] else 0.0

            history_rewards.append(episode_reward)
            history_steps.append(episode_steps)
            history_keys.append(episode_key_acquired)
            history_successes.append(episode_success)

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

            if episode % 100 == 0:
                f.flush()

            if episode == 1 or episode % max(1, args.progress_every) == 0 or episode == num_episodes:
                elapsed = time.perf_counter() - train_start_time
                avg_episode_time = elapsed / episode
                remaining_episodes = num_episodes - episode
                eta_seconds = avg_episode_time * remaining_episodes
                progress_pct = (episode / num_episodes) * 100.0
                print(
                    f"[Progress] {episode}/{num_episodes} ({progress_pct:.1f}%) | "
                    f"elapsed={elapsed / 60:.1f}m | eta={eta_seconds / 60:.1f}m | "
                    f"last_reward={episode_reward:.2f} | success={int(episode_success)}"
                )

            if current_info["is_success"] and episode_positions is not None:
                successful_episodes.append(
                    {
                        "episode": episode,
                        "maze_size": maze_size,
                        "maze_grid": np.asarray(maze_grid, dtype=np.int8),
                        "reward": episode_reward,
                        "steps": episode_steps,
                        "positions": tuple(episode_positions),
                        "key_pick_frame_idx": key_pick_frame_idx,
                    }
                )

            if episode % 500 == 0 and episode_positions is not None:
                save_trajectory_gif(
                    reconstruct_episode_frames(maze_grid, episode_positions, key_pick_frame_idx),
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
        successful_episodes.sort(key=lambda item: (-item["reward"], item["steps"], item["episode"]))
        best_successes = successful_episodes[: args.top_success_gifs]
        print(f"\nSaving top {len(best_successes)} successful trajectory GIFs...")
        for rank, result in enumerate(best_successes, start=1):
            save_trajectory_gif(
                reconstruct_episode_frames(
                    result["maze_grid"],
                    result["positions"],
                    result["key_pick_frame_idx"],
                ),
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

    plot_learning_curves(history_rewards, history_steps, history_successes, plot_filename, output_dir=plots_dir)
    save_checkpoint(checkpoint_filename, network, args, num_episodes)
    print(f"\nTraining completed. Learning curves were generated (output path: {plot_filepath}).")
    print(f"Model checkpoint saved to: {checkpoint_filename}")
