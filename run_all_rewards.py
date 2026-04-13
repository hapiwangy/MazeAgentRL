"""Run training for both A2C and REINFORCE across all supported reward modes.

By default, this script will:
- Enumerate reward modes from `reward_config.get_reward_mode_choices()`.
- Run A2C and REINFORCE sequentially for each reward mode.
- Skip reward modes that depend on LLM rewards if `OPENAI_API_KEY` is not set.

Examples:
  python3 run_all_rewards.py
  python3 run_all_rewards.py --episodes 8000 --maze_size 9 --lr 0.001
  python3 run_all_rewards.py --include_llm
  python3 run_all_rewards.py --dry_run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Iterable, List

from reward_config import get_reward_mode_choices, reward_mode_uses_llm


def _iter_reward_modes(include_llm: bool) -> List[str]:
    modes = list(get_reward_mode_choices())

    if include_llm:
        return modes

    api_key_present = bool(os.getenv("OPENAI_API_KEY"))
    if api_key_present:
        return modes

    return [m for m in modes if not reward_mode_uses_llm(m)]


def _run(cmd: List[str], dry_run: bool) -> None:
    printable = " ".join(cmd)
    print(f"\n=== Running: {printable}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run A2C and REINFORCE for all reward-mode combinations.",
    )

    # Mirrors the defaults you provided in your commands.
    parser.add_argument("--maze_size", type=str, default="9", choices=["9", "25"])
    parser.add_argument("--dataset", type=str, default="dataset/train.json")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--entropy_coef", type=float, default=0.05)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--episodes", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--a2c_run_name",
        type=str,
        default="a2c_sd",
        help="Base run_name for A2C runs (reward mode will be appended).",
    )
    parser.add_argument(
        "--reinforce_run_name",
        type=str,
        default="reinforce",
        help="Base run_name for REINFORCE runs (reward mode will be appended).",
    )

    parser.add_argument(
        "--include_llm",
        action="store_true",
        help=(
            "Include LLM-dependent reward modes even if OPENAI_API_KEY is not set. "
            "(If the key is missing, those runs will likely fail.)"
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without running them.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.include_llm and not os.getenv("OPENAI_API_KEY"):
        print(
            "ERROR: --include_llm was provided but OPENAI_API_KEY is not set in this shell.\n"
            "Set it and re-run, e.g.:\n"
            "  export OPENAI_API_KEY=...\n"
            "  python3 run_all_rewards.py --include_llm\n",
            file=sys.stderr,
        )
        return 2

    reward_modes = _iter_reward_modes(include_llm=args.include_llm)
    skipped_llm = [m for m in get_reward_mode_choices() if reward_mode_uses_llm(m) and m not in reward_modes]

    print("Reward modes to run:")
    for mode in reward_modes:
        print(f"- {mode}")

    if skipped_llm:
        print("\nSkipping LLM-dependent reward modes (OPENAI_API_KEY not set):")
        for mode in skipped_llm:
            print(f"- {mode}")

    # Use the current interpreter for consistency with venv/conda.
    py = sys.executable

    for reward_mode in reward_modes:
        a2c_cmd = [
            py,
            "main.py",
            "--algo",
            "A2C",
            "--maze_size",
            args.maze_size,
            "--dataset",
            args.dataset,
            "--lr",
            str(args.lr),
            "--entropy_coef",
            str(args.entropy_coef),
            "--max_steps",
            str(args.max_steps),
            "--episodes",
            str(args.episodes),
            "--seed",
            str(args.seed),
            "--reward_mode",
            reward_mode,
            "--run_name",
            f"{args.a2c_run_name}_{reward_mode}",
        ]
        _run(a2c_cmd, dry_run=args.dry_run)

        reinforce_cmd = [
            py,
            "main.py",
            "--algo",
            "REINFORCE",
            "--maze_size",
            args.maze_size,
            "--dataset",
            args.dataset,
            "--lr",
            str(args.lr),
            "--entropy_coef",
            str(args.entropy_coef),
            "--max_steps",
            str(args.max_steps),
            "--episodes",
            str(args.episodes),
            "--seed",
            str(args.seed),
            "--reward_mode",
            reward_mode,
            "--run_name",
            f"{args.reinforce_run_name}_{reward_mode}",
        ]
        _run(reinforce_cmd, dry_run=args.dry_run)

    print("\nAll runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
