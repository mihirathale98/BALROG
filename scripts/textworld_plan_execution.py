#!/usr/bin/env python
"""
Plan Execution Script for TextWorld Game Instances.

Takes a JSONL file with generated plans (from textworld_plan_generation.py) and runs
BALROG episodes guided by each plan. Each plan is matched with its original game
instance using the same seed.

Usage:
    # Basic usage
    python scripts/textworld_plan_execution.py \
        --plans-file results/textworld_plans.jsonl \
        --output results/textworld_results.jsonl

    # With local model
    python scripts/textworld_plan_execution.py \
        --plans-file results/textworld_plans.jsonl \
        --model qwen2.5-72b-instruct \
        --base-url http://localhost:8000/v1 \
        --output results/textworld_results.jsonl
"""

import argparse
import copy
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tqdm import tqdm

# Add BALROG to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from balrog.agents import AgentFactory
from balrog.environments import make_env
from balrog.environments.textworld import get_instruction_prompt_with_plan
from balrog.utils import get_unique_seed


def get_power_of_2_budgets(n: int) -> list[int]:
    """Get all powers of 2 from 1 up to n."""
    budgets = [1]
    power = 1
    while 2**power <= n:
        budgets.append(2**power)
        power += 1
    return budgets


def create_config(
    model_id: str,
    base_url: str,
    api_key: str | None = None,
    temperature: float = 1.0,
    max_tokens: int = 4096,
):
    """Create a BALROG-compatible config."""
    config = OmegaConf.create({
        "agent": {
            "type": "naive",
            "remember_cot": True,
            "max_text_history": 16,
            "max_image_history": 0,
            "max_cot_history": 1,
            "max_icl_history": 1000,
            "cache_icl": False,
        },
        "eval": {
            "output_dir": "results",
            "resume_from": None,
            "num_workers": 1,
            "num_episodes": {"textworld": 1},
            "max_steps_per_episode": None,
            "save_trajectories": True,
            "save_images": False,
            "icl_episodes": 1,
            "icl_dataset": "records",
            "feedback_on_invalid_action": True,
        },
        "client": {
            "client_name": "openai",
            "model_id": model_id,
            "base_url": base_url,
            "generate_kwargs": {
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            "thinking_budget": None,
            "timeout": 60,
            "max_retries": 5,
            "delay": 2,
            "alternate_roles": False,
        },
        "envs": {
            "names": "textworld",
            "env_kwargs": {"seed": None},
            "textworld_kwargs": {
                "objective": True,
                "description": True,
                "score": True,
                "max_score": True,
                "won": True,
                "max_episode_steps": 80,
                "textworld_games_path": "tw_games",
            },
        },
        "tasks": {
            "textworld_tasks": ["treasure_hunter", "the_cooking_game", "coin_collector"],
        },
    })

    # Set API key in environment if provided
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    return config


def run_episode_with_plan(
    task: str,
    plan: str,
    config,
    agent_factory,
    episode_idx: int = 0,
    seed: int | None = None,
):
    """Run a single episode with a plan injected into the instruction prompt."""
    # Create environment
    env = make_env("textworld", task, config)

    # Create agent
    agent = agent_factory.create_agent()
    agent.reset()

    # Set seed
    if seed is None:
        seed = get_unique_seed(episode_idx=episode_idx)
    random.seed(seed)
    np.random.seed(seed)

    # Reset environment
    obs, info = env.reset(seed=seed)

    # Get instruction prompt augmented with plan
    plan_augmented_instruction = get_instruction_prompt_with_plan(task, plan)
    agent.prompt_builder.update_instruction_prompt(plan_augmented_instruction)

    # Run episode
    episode_return = 0.0
    max_steps = env.max_steps if config.eval.max_steps_per_episode is None else config.eval.max_steps_per_episode
    actions = []
    done = False

    for step in range(max_steps):
        response = agent.act(obs, prev_action=actions[-1] if actions else None)
        action = env.check_action_validity(response.completion)
        actions.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_return += reward

        # Give feedback on invalid action
        if action != response.completion and config.eval.feedback_on_invalid_action:
            obs["text"]["long_term_context"] = (
                f"\n\nYour previous output did not contain a valid action. Defaulted to action: {action}\n\nObservation:\n"
                + obs["text"]["long_term_context"]
            )

        if done:
            break

    # Collect episode stats
    episode_log = {
        "task": task,
        "plan": plan,
        "episode_return": episode_return,
        "num_steps": step + 1,
        "done": done,
        "won": info.get("won", episode_return > 0),
        "seed": seed,
        "actions": actions,
    }
    episode_log.update(env.get_stats())

    return episode_log


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Execute TextWorld episodes guided by plans"
    )

    # Input/output arguments
    parser.add_argument(
        "--plans-file",
        type=str,
        required=True,
        help="Path to JSONL file with generated plans",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path for results",
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name for episode execution",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8080/v1",
        help="API base URL",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (default: from OPENAI_API_KEY env var)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for agent generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max tokens per agent response",
    )

    # Task arguments
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Specific tasks to run (default: all from plans file)",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=None,
        help="Specific budget levels to test (default: all from plans file)",
    )

    parser.add_argument(
        "--force-run",
        action="store_true",
        help="Re-run even if results exist",
    )

    args = parser.parse_args()

    load_dotenv()

    # Determine API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")

    print(f"Base URL: {args.base_url}")
    print(f"Model: {args.model}")
    print(f"Episodes per plan: {args.episodes_per_plan}")

    # Load plans
    print(f"\nLoading plans from {args.plans_file}...")
    plans_df = pd.read_json(args.plans_file, orient="records", lines=True)
    print(f"Loaded plans for {len(plans_df)} tasks")

    # Determine budget levels from column names
    budget_cols = [col for col in plans_df.columns if col.startswith("bo") and col.endswith("_plan")]
    all_budgets = sorted([int(col.replace("bo", "").replace("_plan", "")) for col in budget_cols])

    budgets = args.budgets if args.budgets else all_budgets
    tasks = args.tasks if args.tasks else plans_df["task"].tolist()

    print(f"Tasks: {tasks}")
    print(f"Budget levels: {budgets}")

    # Create config and agent factory
    config = create_config(
        model_id=args.model,
        base_url=args.base_url,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    agent_factory = AgentFactory(config)

    # Load existing results if available
    output_path = Path(args.output)
    existing_results = []
    if output_path.exists() and not args.force_run:
        print(f"Loading existing results from {output_path}...")
        existing_df = pd.read_json(output_path, orient="records", lines=True)
        existing_results = existing_df.to_dict("records")
        # Create set of completed (task, seed, budget) tuples
        completed = {(r["task"], r["seed"], r["budget"]) for r in existing_results}
    else:
        completed = set()

    # Run episodes - each plan row has a specific seed from plan generation
    results = existing_results.copy()

    # Count total runs: each row in plans_df has a (task, seed) pair
    # For each row, we run len(budgets) episodes (one per budget level)
    filtered_plans = plans_df[plans_df["task"].isin(tasks)]
    total_runs = len(filtered_plans) * len(budgets)
    skipped = 0

    with tqdm(total=total_runs, desc="Running episodes") as pbar:
        for _, row in filtered_plans.iterrows():
            task = row["task"]
            seed = row["seed"]  # Use the same seed from plan generation

            for budget in budgets:
                plan_col = f"bo{budget}_plan"
                if plan_col not in row:
                    pbar.update(1)
                    continue

                plan = row[plan_col]

                # Skip if already completed (keyed by task, seed, budget)
                if (task, seed, budget) in completed:
                    skipped += 1
                    pbar.update(1)
                    continue

                try:
                    episode_log = run_episode_with_plan(
                        task=task,
                        plan=plan,
                        config=config,
                        agent_factory=agent_factory,
                        episode_idx=0,
                        seed=seed,  # Use the same seed as plan generation
                    )

                    result = {
                        "task": task,
                        "seed": seed,
                        "budget": budget,
                        "plan": plan,
                        "episode_return": episode_log["episode_return"],
                        "num_steps": episode_log["num_steps"],
                        "done": episode_log["done"],
                        "won": episode_log["won"],
                    }
                    results.append(result)

                    # Save incrementally
                    df = pd.DataFrame(results)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_json(output_path, orient="records", lines=True)

                    pbar.set_postfix({
                        "task": task[:10],
                        "seed": seed,
                        "bo": budget,
                        "ret": f"{episode_log['episode_return']:.1f}",
                        "won": episode_log["won"],
                    })

                except Exception as e:
                    print(f"\nError running {task} seed={seed} bo{budget}: {e}")
                    results.append({
                        "task": task,
                        "seed": seed,
                        "budget": budget,
                        "plan": plan,
                        "error": str(e),
                    })

                pbar.update(1)

    if skipped > 0:
        print(f"\nSkipped {skipped} already completed episodes")

    # Final save
    df = pd.DataFrame(results)
    df.to_json(output_path, orient="records", lines=True)

    # Compute and print summary
    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY (per task, aggregated across seeds)")
    print("=" * 60)

    # Filter out errors
    df_valid = df[~df.get("error", pd.Series([None] * len(df))).notna()]

    for task in tasks:
        task_df = df_valid[df_valid["task"] == task]
        if len(task_df) == 0:
            continue

        print(f"\n{task}:")
        for budget in budgets:
            budget_df = task_df[task_df["budget"] == budget]
            if len(budget_df) == 0:
                continue

            win_rate = budget_df["won"].mean() if "won" in budget_df.columns else 0
            avg_return = budget_df["episode_return"].mean() if "episode_return" in budget_df.columns else 0
            n_instances = len(budget_df)
            n_wins = budget_df["won"].sum() if "won" in budget_df.columns else 0

            print(f"  bo{budget}: win_rate={win_rate:.2%} ({int(n_wins)}/{n_instances}) avg_return={avg_return:.2f}")

    # Per-seed breakdown
    print("\n" + "=" * 60)
    print("PER-INSTANCE RESULTS (by seed)")
    print("=" * 60)

    for task in tasks:
        task_df = df_valid[df_valid["task"] == task]
        if len(task_df) == 0:
            continue

        seeds = task_df["seed"].unique()
        print(f"\n{task}:")
        for seed in sorted(seeds):
            seed_df = task_df[task_df["seed"] == seed]
            print(f"  seed={seed}:")
            for budget in budgets:
                budget_df = seed_df[seed_df["budget"] == budget]
                if len(budget_df) == 0:
                    continue
                row = budget_df.iloc[0]
                won = row.get("won", False)
                ret = row.get("episode_return", 0)
                print(f"    bo{budget}: won={won} return={ret:.1f}")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
