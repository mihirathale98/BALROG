#!/usr/bin/env python
"""
Plan Generation Script for TextWorld Game Instances (Async).

For each TextWorld task, creates multiple game instances (with different seeds),
gets the initial observation, and generates N plans based on that specific game state.

Uses async generation for efficient vLLM utilization.

Usage:
    # Local vLLM endpoint
    python scripts/textworld_plan_generation.py --local --endpoint http://localhost:8000/v1 \
        --model Qwen/Qwen3-4B-Instruct-2507 --n-plans 8 --n-instances 10
"""

import argparse
import asyncio
import os
import random
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tqdm.asyncio import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "its_hub"))

from its_hub.algorithms import BestOfN
from its_hub.integration.reward_hub import LLMJudgeRewardModel
from its_hub.lms import OpenAICompatibleLanguageModel
from its_hub.utils import extract_content_from_lm_response

from balrog.environments import make_env
from balrog.environments.textworld import intruction_prompts

import litellm
import pandas as pd

litellm.drop_params = True


# Plan generation system prompt for TextWorld
PLAN_GENERATION_SYSTEM_PROMPT = """You are a strategic planner for text adventure games.

You will be given:
1. The game rules and available commands
2. The initial observation showing your starting location and surroundings

Based on this information, create a step-by-step strategy to solve the game.
Focus on:
- What to explore first
- What items to look for and collect
- How to handle obstacles (locked doors, containers, etc.)
- The sequence of actions to achieve the goal

Be specific to what you observe in the game, not generic advice."""


def get_power_of_2_budgets(n: int) -> list[int]:
    """Get all powers of 2 from 1 up to n."""
    budgets = [1]
    power = 1
    while 2**power <= n:
        budgets.append(2**power)
        power += 1
    return budgets


def create_config():
    """Create a minimal BALROG config for environment creation."""
    return OmegaConf.create({
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


def get_initial_observation(task: str, seed: int, config) -> dict:
    """Get the initial observation from a TextWorld game instance."""
    env = make_env("textworld", task, config)

    random.seed(seed)
    np.random.seed(seed)
    obs, info = env.reset(seed=seed)

    # Extract text observation
    long_term = obs["text"].get("long_term_context", "")
    short_term = obs["text"].get("short_term_context", "")

    return {
        "long_term_context": long_term,
        "short_term_context": short_term,
        "full_observation": f"{long_term}\n\n{short_term}".strip(),
    }


async def generate_plans_for_instance(
    task: str,
    seed: int,
    obs: dict,
    task_instruction: str,
    bon: BestOfN,
    plan_lm: OpenAICompatibleLanguageModel,
    n_plans: int,
    budgets: list[int],
) -> dict:
    """Generate plans for a single game instance asynchronously."""
    prompt = f"""GAME RULES AND COMMANDS:
{task_instruction}

INITIAL OBSERVATION:
{obs['full_observation']}

Based on what you see in this specific game, create a detailed step-by-step strategy to win.
Consider the rooms, exits, and objects mentioned in the observation."""

    # Generate all N plans and score them
    result = await bon.ainfer(
        plan_lm, prompt, budget=n_plans, return_response_only=False
    )

    plans = [extract_content_from_lm_response(r) for r in result.responses]
    scores = result.scores

    result_row = {
        "task": task,
        "seed": seed,
        "initial_observation": obs['full_observation'],
        "task_instruction": task_instruction,
        "all_plans": plans,
        "all_scores": scores,
    }

    # Save best plan at each budget level
    for budget in budgets:
        subset_scores = scores[:budget]
        best_idx = subset_scores.index(max(subset_scores))
        result_row[f"bo{budget}_plan"] = plans[best_idx]
        result_row[f"bo{budget}_plan_score"] = scores[best_idx]

    return result_row


async def main_async(args):
    """Async main function."""
    load_dotenv()

    # Determine API key
    if args.local:
        api_key = "NO_API_KEY"
    elif args.api_key:
        api_key = args.api_key
    else:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key and not args.local:
        raise ValueError(
            "API key required. Set OPENAI_API_KEY or use --api-key or --local"
        )

    # Default judge model to same as generation model
    judge_model = args.judge_model or args.model

    print(f"Endpoint: {args.endpoint}")
    print(f"Model: {args.model}")
    print(f"Judge model: {judge_model}")
    print(f"Local mode: {args.local}")
    print(f"Tasks: {args.tasks}")
    print(f"Instances per task: {args.n_instances}")
    print(f"Plans per instance: {args.n_plans}")
    print(f"Max concurrency: {args.max_concurrency}")

    # Set up LM for plan generation (async mode)
    plan_lm = OpenAICompatibleLanguageModel(
        endpoint=args.endpoint,
        api_key=api_key,
        model_name=args.model,
        system_prompt=PLAN_GENERATION_SYSTEM_PROMPT,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_concurrency=args.max_concurrency,
        is_async=True,
    )

    # Set up LLM judge as plan critic
    plan_critic = LLMJudgeRewardModel(
        model=judge_model,
        criterion="overall_quality",
        judge_type="pointwise",
        api_key=api_key,
        base_url=args.endpoint if args.local else None,
        temperature=0.0,
    )

    bon = BestOfN(orm=plan_critic)

    budgets = get_power_of_2_budgets(args.n_plans)
    total_instances = len(args.tasks) * args.n_instances
    print(f"Total game instances: {total_instances}")
    print(f"Budget levels: {budgets}")

    # Create BALROG config for environment
    config = create_config()

    # Prepare all instances
    instances = []
    seeds = list(range(args.start_seed, args.start_seed + args.n_instances))

    print("\nCollecting initial observations...")
    for task in args.tasks:
        task_instruction = intruction_prompts[task].strip()
        for seed in seeds:
            try:
                obs = get_initial_observation(task, seed, config)
                instances.append({
                    "task": task,
                    "seed": seed,
                    "obs": obs,
                    "task_instruction": task_instruction,
                })
            except Exception as e:
                print(f"Error getting observation for {task} seed={seed}: {e}")

    print(f"Collected {len(instances)} instances")

    # Generate plans concurrently with semaphore for rate limiting
    semaphore = asyncio.Semaphore(args.max_concurrency)

    async def generate_with_semaphore(instance):
        async with semaphore:
            try:
                return await generate_plans_for_instance(
                    task=instance["task"],
                    seed=instance["seed"],
                    obs=instance["obs"],
                    task_instruction=instance["task_instruction"],
                    bon=bon,
                    plan_lm=plan_lm,
                    n_plans=args.n_plans,
                    budgets=budgets,
                )
            except Exception as e:
                print(f"\nError generating plans for {instance['task']} seed={instance['seed']}: {e}")
                return None

    print("\nGenerating plans asynchronously...")
    tasks = [generate_with_semaphore(inst) for inst in instances]
    results = await tqdm.gather(*tasks, desc="Generating plans")

    # Filter out None results
    results = [r for r in results if r is not None]

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_json(output_path, orient="records", lines=True)
    print(f"\nSaved {len(results)} game instance plans to {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("PLAN GENERATION SUMMARY")
    print("=" * 60)
    for task in args.tasks:
        task_results = [r for r in results if r["task"] == task]
        print(f"\n{task}: {len(task_results)} instances")
        for budget in budgets:
            if task_results:
                avg_score = np.mean([r[f"bo{budget}_plan_score"] for r in task_results])
                print(f"  bo{budget}: avg_score={avg_score:.3f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate plans for TextWorld game instances using Best-of-N (async)"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model name for plan generation",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Model name for plan evaluation (default: same as --model)",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="https://api.openai.com/v1",
        help="API endpoint URL",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local endpoint (sets api_key to NO_API_KEY)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (default: from OPENAI_API_KEY env var)",
    )

    # Generation arguments
    parser.add_argument(
        "--n-plans",
        type=int,
        default=8,
        help="Number of plans to generate per game instance",
    )
    parser.add_argument(
        "--n-instances",
        type=int,
        default=10,
        help="Number of game instances (seeds) per task (default matches BALROG baseline)",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=42,
        help="Starting seed for game instances",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature for plan generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens per generation",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=32,
        help="Max concurrent API requests",
    )

    # Task arguments
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["treasure_hunter", "the_cooking_game", "coin_collector"],
        help="TextWorld tasks to generate plans for",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default="results/textworld_plans.jsonl",
        help="Output file path",
    )

    args = parser.parse_args()

    # Run async main
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
