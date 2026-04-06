import os
import random

import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch


def resolve_output_path(filename, output_dir):
    """Return a writable path for outputs and ensure the parent directory exists."""
    if os.path.isabs(filename) or os.path.dirname(filename):
        filepath = filename
    else:
        filepath = os.path.join(output_dir, filename)

    parent_dir = os.path.dirname(filepath)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    return filepath


def set_global_seed(seed):
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_method_dirs(base_dir, method_name):
    """Create and return a method-specific output directory."""
    method_dir = os.path.join(base_dir, method_name)
    os.makedirs(method_dir, exist_ok=True)
    return method_dir


def save_trajectory_gif(frames, episode, filename="trajectory.gif", fps=50, output_dir="gifs"):
    """
    Save an episode trajectory as an animated GIF.

    The fps argument controls playback speed. Lower values make the trajectory easier to inspect.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")

    cmap = mcolors.ListedColormap(["white", "black", "blue", "green", "gold", "red"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(frames[0], cmap=cmap, norm=norm)
    total_steps = max(0, len(frames) - 1)
    title = ax.set_title("", fontsize=14)

    def _build_title(frame_idx):
        current_step = min(frame_idx, total_steps)
        status = "Finished" if frame_idx == len(frames) - 1 else "Running"
        return f"Episode {episode} Trajectory | Step {current_step}/{total_steps} | {status}"

    title.set_text(_build_title(0))

    def update(frame_idx):
        im.set_array(frames[frame_idx])
        title.set_text(_build_title(frame_idx))
        return [im, title]

    interval = int(1000 / fps)
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=True)

    filepath = resolve_output_path(filename, output_dir)
    ani.save(filepath, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"[*] Saved trajectory GIF to {filepath} ({fps} FPS)")


def plot_learning_curves(rewards, steps, successes, plot_filename, window=50, output_dir="plots"):
    """
    Plot smoothed learning curves for reward, episode length, and success rate.
    """

    def moving_average(data, w):
        if len(data) < w:
            return data
        return np.convolve(data, np.ones(w), "valid") / w

    smoothed_rewards = moving_average(rewards, window)
    smoothed_steps = moving_average(steps, window)
    smoothed_successes = moving_average(successes, window)

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    axs[0].plot(smoothed_rewards, color="blue")
    axs[0].set_title(f"Episode Return (Moving Average, w={window})")
    axs[0].set_ylabel("Total Reward")

    axs[1].plot(smoothed_steps, color="orange")
    axs[1].set_title(f"Steps to Exit (Moving Average, w={window})")
    axs[1].set_ylabel("Steps")

    axs[2].plot(smoothed_successes, color="green")
    axs[2].set_title(f"Exit Success Rate (Moving Average, w={window})")
    axs[2].set_ylabel("Success Rate")
    axs[2].set_xlabel("Episodes")

    plt.tight_layout()
    filepath = resolve_output_path(plot_filename, output_dir)
    plt.savefig(filepath)
    plt.close(fig)
    print(f"[*] Saved learning curves to {filepath}")
