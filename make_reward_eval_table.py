"""Evaluate checkpoints across reward modes and print a summary table.

This script:
- Enumerates reward modes from reward_config.
- For each reward_mode and algorithm (A2C, REINFORCE), finds the latest matching checkpoint.
- Runs run_test_all.py on that checkpoint.
- Parses the printed summary metrics and prints a markdown table.

Usage:
  python3 make_reward_eval_table.py
  python3 make_reward_eval_table.py --maze_size 9
  python3 make_reward_eval_table.py --deterministic
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from reward_config import get_reward_mode_choices


ALGOS = ("A2C", "REINFORCE")


@dataclass
class Metrics:
    success_rate_pct: float
    avg_steps: float
    avg_sparse_reward: float
    avg_steps_on_success: Optional[float]


_RX_SUCCESS = re.compile(r"^Success rate:\s*([0-9.]+)%\s*$", re.M)
_RX_STEPS = re.compile(r"^Average steps:\s*([0-9.]+)\s*$", re.M)
_RX_REWARD = re.compile(r"^Average sparse reward:\s*([0-9.\-]+)\s*$", re.M)
_RX_SUCCESS_STEPS = re.compile(r"^Average steps on success:\s*([0-9.]+|N/A)\s*$", re.M)


def _latest_checkpoint(algo: str, reward_mode: str) -> Optional[str]:
    # Filenames include: ..._{algo}_{reward_mode}_size{maze_size}_...
    # Use the '_{reward_mode}_size' token to avoid prefix collisions
    # (e.g., reward_mode='sparse' accidentally matching 'sparse_dense').
    pattern = os.path.join("checkpoints", algo, f"*_{algo}_{reward_mode}_size*.pt")
    paths = glob.glob(pattern)
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)


def _parse_metrics(stdout: str) -> Metrics:
    m_success = _RX_SUCCESS.search(stdout)
    m_steps = _RX_STEPS.search(stdout)
    m_reward = _RX_REWARD.search(stdout)
    m_success_steps = _RX_SUCCESS_STEPS.search(stdout)

    if not (m_success and m_steps and m_reward and m_success_steps):
        raise ValueError("Could not parse metrics from run_test_all.py output")

    success_rate_pct = float(m_success.group(1))
    avg_steps = float(m_steps.group(1))
    avg_sparse_reward = float(m_reward.group(1))

    success_steps_raw = m_success_steps.group(1)
    avg_steps_on_success = None if success_steps_raw == "N/A" else float(success_steps_raw)

    return Metrics(
        success_rate_pct=success_rate_pct,
        avg_steps=avg_steps,
        avg_sparse_reward=avg_sparse_reward,
        avg_steps_on_success=avg_steps_on_success,
    )


def _eval_checkpoint(checkpoint_path: str, deterministic: bool, maze_size: Optional[int]) -> Metrics:
    cmd: List[str] = [sys.executable, "run_test_all.py", "--checkpoint", checkpoint_path]
    if deterministic:
        cmd.append("--deterministic")
    if maze_size is not None:
        cmd.extend(["--maze_size", str(maze_size)])

    p = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return _parse_metrics(p.stdout)


def _fmt_cell(metrics: Optional[Metrics]) -> str:
    if metrics is None:
        return ""
    lines = [
        f"Success rate: {metrics.success_rate_pct:.2f}%",
        f"Average steps: {metrics.avg_steps:.2f}",
        f"Average sparse reward: {metrics.avg_sparse_reward:.2f}",
    ]
    if metrics.avg_steps_on_success is None:
        lines.append("Average steps on success: N/A")
    else:
        lines.append(f"Average steps on success: {metrics.avg_steps_on_success:.2f}")
    return "<br>".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate latest checkpoints and print a summary table.")
    parser.add_argument("--maze_size", type=int, default=None, help="Optional maze size filter for evaluation.")
    parser.add_argument("--deterministic", action="store_true", help="Use greedy evaluation policy.")
    args = parser.parse_args()

    reward_modes = list(get_reward_mode_choices())

    table: Dict[Tuple[str, str], Optional[Metrics]] = {}

    for reward_mode in reward_modes:
        for algo in ALGOS:
            ckpt = _latest_checkpoint(algo, reward_mode)
            if ckpt is None:
                table[(reward_mode, algo)] = None
                continue
            print(f"Evaluating {algo} / {reward_mode}: {ckpt}")
            table[(reward_mode, algo)] = _eval_checkpoint(
                ckpt,
                deterministic=args.deterministic,
                maze_size=args.maze_size,
            )

    print("\n| reward_mode | A2C | REINFORCE |")
    print("|---|---|---|")
    for reward_mode in reward_modes:
        a2c_cell = _fmt_cell(table[(reward_mode, "A2C")])
        rein_cell = _fmt_cell(table[(reward_mode, "REINFORCE")])
        print(f"| {reward_mode} | {a2c_cell} | {rein_cell} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
