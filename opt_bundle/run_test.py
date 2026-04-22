import argparse
import glob
import os
import subprocess
import sys


def find_latest_checkpoint(checkpoint_dir):
    pattern = os.path.join(checkpoint_dir, "*", "*.pt")
    checkpoint_paths = glob.glob(pattern)
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoint files were found under {checkpoint_dir}.")
    return max(checkpoint_paths, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation on dataset/test.json using a trained checkpoint."
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
        "--maze_id",
        type=str,
        default=None,
        help="Optional maze id from the test dataset.",
    )
    parser.add_argument(
        "--maze_size",
        type=int,
        default=None,
        help="Optional maze size override.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional max-steps override.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional evaluation seed override.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="eval_test",
        help="Tag for evaluation outputs.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use greedy argmax action selection during evaluation.",
    )
    parser.add_argument(
        "--save_gif",
        action="store_true",
        help="Save the evaluation trajectory as a GIF.",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or find_latest_checkpoint("checkpoints")
    command = [
        sys.executable,
        "test_agent.py",
        "--checkpoint",
        checkpoint_path,
        "--dataset",
        args.dataset,
        "--run_name",
        args.run_name,
    ]

    if args.maze_id is not None:
        command.extend(["--maze_id", str(args.maze_id)])
    if args.maze_size is not None:
        command.extend(["--maze_size", str(args.maze_size)])
    if args.max_steps is not None:
        command.extend(["--max_steps", str(args.max_steps)])
    if args.seed is not None:
        command.extend(["--seed", str(args.seed)])
    if args.deterministic:
        command.append("--deterministic")
    if args.save_gif:
        command.append("--save_gif")

    print(f"Using checkpoint: {checkpoint_path}")
    print("Running command:")
    print(" ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
