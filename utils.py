import os

import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def save_trajectory_gif(frames, episode, filename="trajectory.gif", fps=50):
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
    ax.set_title(f"Episode {episode} Trajectory", fontsize=14)

    def update(frame_idx):
        im.set_array(frames[frame_idx])
        return [im]

    interval = int(1000 / fps)
    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=True)

    os.makedirs("gifs", exist_ok=True)
    filepath = os.path.join("gifs", filename)
    ani.save(filepath, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"[*] Saved trajectory GIF to {filepath} ({fps} FPS)")


def plot_learning_curves(rewards, steps, successes, plot_filename, window=50):
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
    os.makedirs("plots", exist_ok=True)
    filepath = os.path.join("plots", plot_filename)
    plt.savefig(filepath)
    plt.close(fig)
    print(f"[*] Saved learning curves to {filepath}")
