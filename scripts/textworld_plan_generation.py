#!/usr/bin/env python
"""
Plan Generation Script for TextWorld Tasks.

Generates N high-level game-solving plans for each TextWorld task using BestOfN,
then saves the best plan at each budget level.

Usage:
    # OpenAI API
    python scripts/textworld_plan_generation.py --n-plans 8 --output results/textworld_plans.jsonl

    # Local vLLM endpoint
    python scripts/textworld_plan_generation.py --local --endpoint http://localhost:8000/v1 \
        --model qwen2.5-72b-instruct --n-plans 8
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

# Add its_hub to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "its_hub"))

from its_hub.algorithms import BestOfN
from its_hub.integration.reward_hub import LLMJudgeRewardModel
from its_hub.lms import OpenAICompatibleLanguageModel
from its_hub.utils import extract_content_from_lm_response

import litellm
import pandas as pd

litellm.drop_params = True


# Plan generation system prompt for TextWorld
PLAN_GENERATION_SYSTEM_PROMPT = """You are a strategic planner for text adventure games.

When given a game description, provide ONLY a high-level strategy or plan to solve the game.
Do NOT provide specific commands - describe the logical approach.

Your plan should include:
1. Key objectives to accomplish
2. The sequence of steps to take (explore, find items, solve puzzles)
3. How to handle common obstacles (locked doors, hidden items, etc.)

Keep the plan concise and focused on strategy, not specific game commands."""


# TextWorld task descriptions (matching balrog/environments/textworld/__init__.py)
TEXTWORLD_TASKS = {
    "treasure_hunter": """
TextWorld: Treasure Hunter

You are in a randomly generated maze with multiple rooms. Your goal is to find a specific treasure object.

Key mechanics:
- Explore different rooms using directional commands
- Look for keys to unlock locked doors and containers
- Keys match locks by their adjective (e.g., "non-euclidean keycard" matches "non-euclidean safe")
- Containers may hold the target object or keys
- You must unlock, then open locked doors/containers

You have 40 steps to find and obtain the treasure.
""",
    "the_cooking_game": """
TextWorld: The Cooking Game

You navigate through rooms to find ingredients, prepare food according to a recipe, and eat the meal.

Key mechanics:
- Find and examine the cookbook to see the recipe
- Gather ingredients from various rooms
- Process ingredients: slice/chop/dice with a knife (take both knife and ingredient first)
- Cook ingredients: BBQ=grill, stove=fry, oven=roast (ingredient must be in inventory, tool in room)
- Process before cooking (e.g., slice then fry)
- Prepare meal in kitchen when all ingredients ready, then eat meal

You have 80 steps to complete the task.
""",
    "coin_collector": """
TextWorld: Coin Collector

You are in a randomly generated maze. Your goal is to find and collect a coin.

Key mechanics:
- Navigate rooms using directional commands (go north/south/east/west)
- Explore until you find the coin
- Take the coin when you see it

You have 25 steps to find the coin.
""",
}


def get_power_of_2_budgets(n: int) -> list[int]:
    """Get all powers of 2 from 1 up to n."""
    budgets = [1]  # Include 1 for baseline
    power = 1
    while 2**power <= n:
        budgets.append(2**power)
        power += 1
    return budgets


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate plans for TextWorld tasks using Best-of-N"
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
        help="Number of plans to generate per task",
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
        default=16,
        help="Max concurrent API requests",
    )

    # Task arguments
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=list(TEXTWORLD_TASKS.keys()),
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

    # Set up LM for plan generation
    plan_lm = OpenAICompatibleLanguageModel(
        endpoint=args.endpoint,
        api_key=api_key,
        model_name=args.model,
        system_prompt=PLAN_GENERATION_SYSTEM_PROMPT,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_concurrency=args.max_concurrency,
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
    print(f"Processing {len(args.tasks)} tasks, {args.n_plans} plans each")
    print(f"Budget levels: {budgets}")

    results = []
    for task in tqdm(args.tasks, desc="Tasks"):
        task_description = TEXTWORLD_TASKS[task]

        # Prompt for plan generation
        prompt = f"""Generate a step-by-step strategy to solve this text adventure game:

{task_description}

Provide a clear, actionable plan that will help an agent complete this game efficiently."""

        # Generate all N plans and score them in one call
        result = bon.infer(
            plan_lm, prompt, budget=args.n_plans, return_response_only=False
        )

        plans = [extract_content_from_lm_response(r) for r in result.responses]
        scores = result.scores

        result_row = {
            "task": task,
            "task_description": task_description.strip(),
            "all_plans": plans,
            "all_scores": scores,
        }

        # Simulate each budget level from the same N plans
        for budget in budgets:
            subset_scores = scores[:budget]
            best_idx = subset_scores.index(max(subset_scores))
            result_row[f"bo{budget}_plan"] = plans[best_idx]
            result_row[f"bo{budget}_plan_score"] = scores[best_idx]

        results.append(result_row)

        # Print best plan for this task
        best_overall_idx = scores.index(max(scores))
        print(f"\n{task}: Best plan (score={scores[best_overall_idx]:.2f}):")
        print("-" * 40)
        print(plans[best_overall_idx][:500] + "..." if len(plans[best_overall_idx]) > 500 else plans[best_overall_idx])

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_json(output_path, orient="records", lines=True)
    print(f"\nSaved {len(results)} task plans to {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("PLAN GENERATION SUMMARY")
    print("=" * 60)
    for row in results:
        print(f"\n{row['task']}:")
        for budget in budgets:
            score = row[f"bo{budget}_plan_score"]
            print(f"  bo{budget}: score={score:.3f}")


if __name__ == "__main__":
    main()
