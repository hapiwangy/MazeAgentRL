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
from OpenAILLM import OpenAILLM
from REINFORCE import REINFORCEAgent, REINFORCENetwork
from RewardEngine import RewardEngine
from utils import plot_learning_curves, save_trajectory_gif


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
        default=1500,
        help="Number of training episodes.",
    )
    parser.add_argument('--maze_size', 
                        type=str, 
                        default='9', 
                        choices=['9', '25'], 
                        help='size of the maze in current setting 9X9 or 25X25')
    args = parser.parse_args()

    # 2. Prepare the CSV log file.
    os.makedirs("logs", exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/run_{args.algo}_size{args.maze_size}_lr{args.lr}_{run_timestamp}.csv"

    with open(log_filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Maze_ID", "Maze_Size", "Steps", "Total_Reward", "Success", "Loss"])

    print(f"Training log will be written to: {log_filename}")

    # 3. Load the maze dataset.
    if not os.path.exists(args.dataset):
            raise FileNotFoundError(f"can not find the dataset {args.dataset}！run MazeGenerator.py first")
    with open(args.dataset, 'r', encoding='utf-8') as f:
        full_dataset = json.load(f)
    target_size = int(args.maze_size)
    maze_dataset = [m for m in full_dataset if m["size"] == target_size]
 
    print(f"load {args.maze_size}x{args.maze_size} numbers of dataset {len(maze_dataset)} for training")

    # 4. Initialize the model, optimizer, and reward modules.
    if args.algo == "A2C":
        network = A2CNetwork()
        agent = A2CAgent(network)
    else:
        network = REINFORCENetwork()
        agent = REINFORCEAgent(network)

    optimizer = optim.Adam(network.parameters(), lr=args.lr)
    reward_engine = RewardEngine()
    llm_api = OpenAILLM(model_name="gpt-4o-mini")

    gamma = 0.99
    num_episodes = args.episodes

    history_rewards = []
    history_steps = []
    history_successes = []

    print(
        f"Starting {args.algo} training for {num_episodes} episodes "
        f"(lr={args.lr}, max_steps={args.max_steps}).\n"
    )
    successful_gifs_saved = 0

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
        reward_engine.reset()

        log_probs, values, rewards, entropies = [], [], [], []
        episode_frames = [env.current_map.copy()]

        done = False
        while not done:
            if args.algo == "A2C":
                action, log_prob, value, entropy = agent.select_action(obs)
                values.append(value)
            else:
                action, log_prob, entropy = agent.select_action(obs)

            next_obs, sparse_reward, terminated, truncated, current_info = env.step(action)
            done = terminated or truncated

            frame = env.current_map.copy()
            frame[current_info["agent_pos"]] = 5
            episode_frames.append(frame)

            dense_reward = reward_engine.compute_dense_reward(current_info, prev_info)
            raw_llm_reward = llm_api.get_reward(current_info, prev_info)
            bounded_llm_reward = reward_engine.apply_llm_bounds(raw_llm_reward)
            total_step_reward = sparse_reward + dense_reward + bounded_llm_reward

            log_probs.append(log_prob)
            rewards.append(total_step_reward)
            entropies.append(entropy)

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
        episode_reward = sum(rewards)
        episode_steps = env.current_step
        episode_success = 1.0 if current_info["is_success"] else 0.0

        history_rewards.append(episode_reward)
        history_steps.append(episode_steps)
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
                    int(episode_success),
                    round(loss.item(), 4),
                ]
            )

        # Save a few successful trajectories for qualitative inspection.
        if current_info["is_success"] and successful_gifs_saved < 3:
            print(
                f"\nEpisode {episode} solved a {maze_size}x{maze_size} maze. "
                "Saving trajectory GIF..."
            )
            save_trajectory_gif(
                episode_frames,
                episode,
                filename=f"traj_{args.algo}_SUCCESS_{maze_size}x{maze_size}_ep{episode}.gif",
                fps=20,
            )
            successful_gifs_saved += 1

        if episode % 500 == 0:
            save_trajectory_gif(
                episode_frames,
                episode,
                filename=f"traj_{args.algo}_{maze_size}x{maze_size}_ep{episode}.gif",
                fps=15,
            )

        if episode % 100 == 0:
            avg_reward = np.mean(history_rewards[-100:])
            avg_success = np.mean(history_successes[-100:])
            print(
                f"Episode {episode:4d} | Avg Reward: {avg_reward:.2f} | "
                f"Success Rate: {avg_success * 100:.1f}% | Loss: {loss.item():.4f} | "
                f"Current Maze Size: {maze_size}x{maze_size}"
            )

    plot_filename = f"plots/learning_curves_{args.algo}_size{args.maze_size}_{run_timestamp}.png"
    plot_learning_curves(history_rewards, history_steps, history_successes, plot_filename)
    print(
        f"\nTraining completed. Learning curves were generated "
        f"(default output path in plots/, run tag: {plot_filename})."
    )
