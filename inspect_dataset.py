import json
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def visualize_dataset(json_path="dataset/train.json", output_dir="dataset_images", num_samples=5):
    """
    Loads a maze dataset from a JSON file, randomly samples a specified number 
    of mazes, and generates high-resolution PNG visualizations for inspection.

    Args:
        json_path (str): The file path to the JSON dataset.
        output_dir (str): The directory where the output images will be saved.
        num_samples (int): The number of random samples to visualize.
    """
    # Verify the existence of the dataset file
    if not os.path.exists(json_path):
        print(f"[Error] Dataset not found at '{json_path}'. "
              f"Please ensure 'MazeGenerator.py' has been executed successfully.")
        return

    # Create the output directory if it does not already exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the JSON dataset
    with open(json_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    
    print(f"[Info] Successfully loaded dataset from '{json_path}'. Total mazes: {len(dataset)}.")
    
    # Randomly sample from the dataset (ensure we do not exceed the dataset size)
    samples = random.sample(dataset, min(num_samples, len(dataset)))

    # Define a color map consistent with the trajectory GIF generation
    # Mapping: 0: Path (White), 1: Wall (Black), 2: Start (Blue), 3: Exit (Green), 4: Key (Gold)
    cmap = mcolors.ListedColormap(['white', 'black', 'blue', 'green', 'gold'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    for index, data in enumerate(samples):
        maze_id = data["id"]
        maze_size = data["size"]
        grid = np.array(data["grid"])

        # Initialize the plot figure
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis('off')  # Hide coordinate axes for a cleaner visualization
        
        # Render the maze grid using the defined color map
        ax.imshow(grid, cmap=cmap, norm=norm)
        
        # Set a descriptive title including the maze ID and dimensions
        ax.set_title(f"Maze ID: {maze_id} ({maze_size}x{maze_size})", fontsize=14)

        # Save the generated figure as a high-resolution PNG
        filepath = os.path.join(output_dir, f"{maze_id}.png")
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close(fig)
        
        print(f"[Success] Saved maze visualization to: {filepath}")

if __name__ == "__main__":
    # Execute the visualization function for both training and validation datasets
    print("=== Initiating Dataset Inspection ===")
    visualize_dataset(json_path="dataset/train.json", output_dir="dataset_images/train", num_samples=5)
    visualize_dataset(json_path="dataset/val.json", output_dir="dataset_images/val", num_samples=3)
    
    print("\n[Complete] Visualization generation finished. Please review the output images in the 'dataset_images' directory.")