#!/usr/bin/env python3
"""
Generate workplace emails using LLMs for gender bias study.

Usage:
    python generate_emails.py --model gpt-5.2 --api-key $OPENAI_API_KEY --output emails.json
    python generate_emails.py --model gemini --api-key $GEMINI_API_KEY --output emails.json

The script includes checkpoint logic - if interrupted, rerun the same command to resume.
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from config import get_all_personas, SCENARIOS, get_generation_prompt, SUPPORTED_MODELS


def call_openai(prompt: str, api_key: str, model: str = "gpt-5.2") -> str:
    """Call OpenAI API."""
    import openai
    client = openai.OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1000,
    )
    return response.choices[0].message.content


def call_gemini(prompt: str, api_key: str, model: str = "gemini-2.0-flash") -> str:
    """Call Google Gemini API."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    
    model_obj = genai.GenerativeModel(model)
    response = model_obj.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=1000,
        )
    )
    return response.text


def generate_email(
    model: str,
    api_key: str,
    persona,
    scenario_id: str,
    response_number: int,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Generate a single email.
    
    Returns dict with email data or error information.
    """
    prompt = get_generation_prompt(persona, scenario_id)
    model_config = SUPPORTED_MODELS[model]
    
    for attempt in range(max_retries):
        try:
            if model_config["provider"] == "openai":
                email_text = call_openai(prompt, api_key, model_config["model_id"])
            else:
                email_text = call_gemini(prompt, api_key, model_config["model_id"])
            
            return {
                "response_id": f"{model}_{persona.id}_{scenario_id}_{response_number:02d}",
                "model": model_config["model_id"],
                "persona_id": persona.id,
                "persona_name": persona.name,
                "gender": persona.gender,
                "scenario_id": scenario_id,
                "scenario_name": SCENARIOS[scenario_id]["name"],
                "response_number": response_number,
                "email_text": email_text,
                "timestamp": datetime.now().isoformat(),
                "error": None,
            }
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"  Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                return {
                    "response_id": f"{model}_{persona.id}_{scenario_id}_{response_number:02d}",
                    "model": model_config["model_id"],
                    "persona_id": persona.id,
                    "persona_name": persona.name,
                    "gender": persona.gender,
                    "scenario_id": scenario_id,
                    "scenario_name": SCENARIOS[scenario_id]["name"],
                    "response_number": response_number,
                    "email_text": None,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                }


def load_checkpoint(output_path: str) -> Dict[str, Any]:
    """Load existing results for checkpoint resume."""
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            data = json.load(f)
        return {item["response_id"]: item for item in data}
    return {}


def save_checkpoint(results: List[Dict], output_path: str):
    """Save current results to file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate workplace emails for gender bias study")
    parser.add_argument("--model", required=True, choices=list(SUPPORTED_MODELS.keys()),
                        help="Model to use for generation")
    parser.add_argument("--api-key", required=True, help="API key for the model")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--responses-per-persona", type=int, default=3,
                        help="Number of responses per persona-scenario combination")
    parser.add_argument("--checkpoint-interval", type=int, default=10,
                        help="Save checkpoint every N responses")
    args = parser.parse_args()
    
    # Load checkpoint if exists
    completed = load_checkpoint(args.output)
    print(f"Loaded checkpoint: {len(completed)} responses already completed")
    
    # Build task list
    personas = get_all_personas()
    tasks = []
    for persona in personas:
        for scenario_id in SCENARIOS.keys():
            for resp_num in range(1, args.responses_per_persona + 1):
                response_id = f"{args.model}_{persona.id}_{scenario_id}_{resp_num:02d}"
                if response_id not in completed:
                    tasks.append((persona, scenario_id, resp_num))
    
    # Shuffle to avoid systematic effects
    random.shuffle(tasks)
    
    print(f"Remaining tasks: {len(tasks)}")
    print(f"Model: {args.model} ({SUPPORTED_MODELS[args.model]['model_id']})")
    
    # Generate emails
    results = list(completed.values())
    
    for i, (persona, scenario_id, resp_num) in enumerate(tasks):
        print(f"[{i+1}/{len(tasks)}] {persona.id} {scenario_id} #{resp_num}...", end=" ")
        
        result = generate_email(
            model=args.model,
            api_key=args.api_key,
            persona=persona,
            scenario_id=scenario_id,
            response_number=resp_num,
        )
        
        results.append(result)
        
        if result["error"]:
            print(f"ERROR: {result['error']}")
        else:
            print("OK")
        
        # Checkpoint
        if (i + 1) % args.checkpoint_interval == 0:
            save_checkpoint(results, args.output)
            print(f"  [Checkpoint saved: {len(results)} total]")
        
        # Rate limiting
        time.sleep(0.5)
    
    # Final save
    save_checkpoint(results, args.output)
    
    # Summary
    errors = sum(1 for r in results if r["error"])
    print(f"\nComplete: {len(results)} responses, {errors} errors")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
