#!/usr/bin/env python3
"""
Configure its_hub IaaS service to use LLM Judge with Best-of-N

This script configures the IaaS service to use an LLM judge (like GPT-4o-mini)
to evaluate and select the best response from multiple candidates.

Usage:
    python scripts/configure_iaas_llm_judge.py
"""

import os
import sys
import requests


def configure_iaas_with_llm_judge(
    iaas_url: str = "http://localhost:8080",
    main_lm_endpoint: str = "http://localhost:8000/v1",
    main_model: str = "your-model-name",
    judge_model: str = "gpt-4o-mini",
    judge_criterion: str = "overall_quality",
    judge_api_key: str | None = None,
    judge_base_url: str | None = None,
):
    """
    Configure IaaS service to use LLM judge with Best-of-N

    Args:
        iaas_url: URL of the IaaS service
        main_lm_endpoint: Endpoint of the main language model (generates candidates)
        main_model: Model name for the main LM
        judge_model: LiteLLM model name for judge (e.g., 'gpt-4o-mini', 'claude-3-sonnet-20240229')
        judge_criterion: Evaluation criterion (overall_quality, technical_quality, etc.)
        judge_api_key: API key for judge model provider
        judge_base_url: Base URL for custom judge endpoint (optional)
    """

    config = {
        # Main LLM configuration (generates candidate responses)
        "endpoint": main_lm_endpoint,
        "api_key": "EMPTY",  # Use actual key if your main LM requires auth
        "model": main_model,

        # Algorithm configuration
        "alg": "best-of-n",

        # LLM Judge configuration
        "use_llm_judge": True,
        "judge_model": judge_model,
        "judge_criterion": judge_criterion,
        "judge_api_key": judge_api_key,
    }

    # Add optional judge base URL if provided
    if judge_base_url:
        config["judge_base_url"] = judge_base_url

    print(f"Configuring IaaS at {iaas_url}/configure")
    print(f"  Main LM: {main_model} @ {main_lm_endpoint}")
    print(f"  Judge: {judge_model} (criterion: {judge_criterion})")
    print()

    try:
        response = requests.post(
            f"{iaas_url}/configure",
            json=config,
            timeout=30
        )

        if response.status_code == 200:
            print("✅ IaaS configured successfully!")
            print(response.json())
            return True
        else:
            print(f"❌ Configuration failed: {response.status_code}")
            print(response.text)
            return False

    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to IaaS at {iaas_url}")
        print("Make sure the IaaS service is running:")
        print("  its-iaas --port 8080")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Main entry point"""

    # Get judge API key from environment
    judge_api_key = os.getenv("OPENAI_API_KEY")
    if not judge_api_key:
        print("⚠️  Warning: OPENAI_API_KEY not set in environment")
        print("Set it with: export OPENAI_API_KEY='your-key'")
        print()
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Configuration presets
    print("Select a configuration preset:")
    print()
    print("1. OpenAI GPT-4o-mini judge (recommended for cost)")
    print("2. Anthropic Claude Sonnet judge")
    print("3. Custom configuration")
    print()

    choice = input("Choice (1-3): ").strip()

    if choice == "1":
        # OpenAI GPT-4o-mini
        success = configure_iaas_with_llm_judge(
            judge_model="gpt-4o-mini",
            judge_criterion="overall_quality",
            judge_api_key=judge_api_key or os.getenv("OPENAI_API_KEY"),
        )
    elif choice == "2":
        # Anthropic Claude
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_key:
            print("❌ ANTHROPIC_API_KEY not set")
            print("Set it with: export ANTHROPIC_API_KEY='your-key'")
            sys.exit(1)

        success = configure_iaas_with_llm_judge(
            judge_model="claude-3-sonnet-20240229",
            judge_criterion="overall_quality",
            judge_api_key=anthropic_key,
        )
    elif choice == "3":
        # Custom configuration
        print("\nCustom Configuration:")
        main_endpoint = input("Main LM endpoint [http://localhost:8000/v1]: ").strip() or "http://localhost:8000/v1"
        main_model = input("Main model name: ").strip()
        judge_model = input("Judge model [gpt-4o-mini]: ").strip() or "gpt-4o-mini"
        judge_criterion = input("Judge criterion [overall_quality]: ").strip() or "overall_quality"

        success = configure_iaas_with_llm_judge(
            main_lm_endpoint=main_endpoint,
            main_model=main_model,
            judge_model=judge_model,
            judge_criterion=judge_criterion,
            judge_api_key=judge_api_key,
        )
    else:
        print("Invalid choice")
        sys.exit(1)

    if success:
        print()
        print("=" * 60)
        print("✨ IaaS is ready to use!")
        print()
        print("Test it with:")
        print("  python scripts/test_iaas_llm_judge.py")
        print()
        print("Or use the OpenAI client:")
        print('  from openai import OpenAI')
        print('  client = OpenAI(base_url="http://localhost:8080/v1", api_key="EMPTY")')
        print('  response = client.chat.completions.create(')
        print('      model="your-model-name",')
        print('      messages=[{"role": "user", "content": "Your prompt"}],')
        print('      extra_body={"budget": 8}')
        print('  )')
        print("=" * 60)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
