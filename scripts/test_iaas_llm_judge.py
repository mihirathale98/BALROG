#!/usr/bin/env python3
"""
Test script for IaaS with LLM Judge

This script tests the configured IaaS service by sending sample requests.

Usage:
    python scripts/test_iaas_llm_judge.py
"""

import time
from openai import OpenAI


def test_iaas_llm_judge(
    iaas_url: str = "http://localhost:8080/v1",
    model: str = "your-model-name",
    budget: int = 5
):
    """Test IaaS with LLM judge using OpenAI client"""

    client = OpenAI(
        base_url=iaas_url,
        api_key="EMPTY"
    )

    test_prompts = [
        {
            "name": "Simple Explanation",
            "prompt": "Explain quantum computing in simple terms.",
            "budget": 5
        },
        {
            "name": "Code Generation",
            "prompt": "Write a Python function to check if a number is prime.",
            "budget": 8
        },
        {
            "name": "Technical Question",
            "prompt": "What are the key differences between REST and GraphQL APIs?",
            "budget": 6
        },
    ]

    print("=" * 70)
    print("🧪 Testing IaaS with LLM Judge")
    print("=" * 70)
    print()

    for i, test in enumerate(test_prompts, 1):
        print(f"Test {i}/{len(test_prompts)}: {test['name']}")
        print(f"Prompt: {test['prompt']}")
        print(f"Budget: {test['budget']} (generating {test['budget']} candidates)")
        print("-" * 70)

        start = time.time()

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": test["prompt"]}
                ],
                extra_body={"budget": test["budget"]}
            )

            elapsed = time.time() - start

            print(f"\n✅ Response (took {elapsed:.2f}s):")
            print(response.choices[0].message.content)

            # Show metadata if available
            if hasattr(response, 'metadata') and response.metadata:
                print(f"\n📊 Metadata:")
                for key, value in response.metadata.items():
                    if key != 'all_responses':  # Don't print all responses
                        print(f"  {key}: {value}")

        except Exception as e:
            print(f"\n❌ Error: {e}")

        print("\n" + "=" * 70 + "\n")


def main():
    """Main entry point"""

    print("Enter your configuration (press Enter for defaults):\n")

    iaas_url = input("IaaS URL [http://localhost:8080/v1]: ").strip()
    if not iaas_url:
        iaas_url = "http://localhost:8080/v1"

    model = input("Model name: ").strip()
    if not model:
        print("❌ Model name is required")
        return

    budget_str = input("Budget (number of candidates) [5]: ").strip()
    budget = int(budget_str) if budget_str else 5

    print()
    test_iaas_llm_judge(
        iaas_url=iaas_url,
        model=model,
        budget=budget
    )


if __name__ == "__main__":
    main()
