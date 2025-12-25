#!/usr/bin/env python3
"""
Rate workplace emails using LLMs for gender bias study.

Usage:
    python rate_emails.py --evaluator gemini --api-key $GEMINI_API_KEY \
        --input emails_gpt52.json --output ratings.json --prompt-style naturalistic
    
    python rate_emails.py --evaluator gpt-5.2 --api-key $OPENAI_API_KEY \
        --input emails_gemini.json --output ratings.json --prompt-style debiased

The script includes checkpoint logic - if interrupted, rerun the same command to resume.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from config import get_evaluation_prompt, SUPPORTED_MODELS


def call_openai(prompt: str, api_key: str, model: str = "gpt-5.2") -> str:
    """Call OpenAI API."""
    import openai
    client = openai.OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,  # Lower temperature for more consistent ratings
        max_tokens=500,
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
            temperature=0.3,
            max_output_tokens=500,
        )
    )
    return response.text


def parse_rating_response(response_text: str, scenario_id: str) -> Dict[str, Any]:
    """
    Parse JSON rating from model response.
    
    Handles cases where model includes markdown code blocks or extra text.
    """
    # Try to extract JSON from response
    text = response_text.strip()
    
    # Remove markdown code blocks if present
    if "```json" in text:
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
    elif "```" in text:
        text = re.sub(r'```\s*', '', text)
    
    # Try to find JSON object
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        text = match.group()
    
    try:
        data = json.loads(text)
        return {"parsed": data, "parse_error": False, "raw_response": response_text}
    except json.JSONDecodeError:
        # Try to extract individual fields with regex
        result = {"parse_error": True, "raw_response": response_text}
        
        if scenario_id == "S01":
            fields = ['likelihood_to_grant_raise', 'professionalism', 
                     'perceived_confidence', 'perceived_competence']
        else:
            fields = ['likelihood_to_send_correction', 'professionalism',
                     'perceived_reasonableness', 'seems_entitled']
        
        for field in fields:
            match = re.search(rf'"{field}":\s*(\d)', response_text)
            if match:
                result[field] = int(match.group(1))
        
        # Boolean fields
        match = re.search(r'"would_push_back":\s*(true|false)', response_text, re.IGNORECASE)
        if match:
            result['would_push_back'] = match.group(1).lower() == 'true'
        
        # String fields
        match = re.search(r'"initial_reaction":\s*"(\w+)"', response_text)
        if match:
            result['initial_reaction'] = match.group(1)
        
        match = re.search(r'"would_affect_perception":\s*"(\w+)"', response_text)
        if match:
            result['would_affect_perception'] = match.group(1)
        
        return result


def rate_email(
    evaluator: str,
    api_key: str,
    email_data: Dict[str, Any],
    prompt_style: str,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Rate a single email.
    
    Returns dict with rating data or error information.
    """
    scenario_id = email_data["scenario_id"]
    email_text = email_data["email_text"]
    debiased = (prompt_style == "debiased")
    
    prompt = get_evaluation_prompt(scenario_id, email_text, debiased=debiased)
    model_config = SUPPORTED_MODELS[evaluator]
    
    for attempt in range(max_retries):
        try:
            if model_config["provider"] == "openai":
                response_text = call_openai(prompt, api_key, model_config["model_id"])
            else:
                response_text = call_gemini(prompt, api_key, model_config["model_id"])
            
            parsed = parse_rating_response(response_text, scenario_id)
            
            result = {
                "response_id": email_data["response_id"],
                "generator_model": email_data["model"],
                "evaluator_model": evaluator,
                "prompt_style": prompt_style,
                "persona_id": email_data["persona_id"],
                "persona_name": email_data["persona_name"],
                "gender": email_data["gender"],
                "scenario_id": scenario_id,
                "scenario_name": email_data["scenario_name"],
                "timestamp": datetime.now().isoformat(),
            }
            
            # Merge parsed ratings
            if parsed.get("parsed"):
                result.update(parsed["parsed"])
            result["parse_error"] = parsed.get("parse_error", False)
            result["raw_response"] = parsed.get("raw_response", "")
            
            # Also include any regex-extracted fields
            for key in parsed:
                if key not in ["parsed", "parse_error", "raw_response"]:
                    result[key] = parsed[key]
            
            return result
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"  Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                return {
                    "response_id": email_data["response_id"],
                    "generator_model": email_data["model"],
                    "evaluator_model": evaluator,
                    "prompt_style": prompt_style,
                    "persona_id": email_data["persona_id"],
                    "persona_name": email_data["persona_name"],
                    "gender": email_data["gender"],
                    "scenario_id": scenario_id,
                    "scenario_name": email_data["scenario_name"],
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
    parser = argparse.ArgumentParser(description="Rate workplace emails for gender bias study")
    parser.add_argument("--evaluator", required=True, choices=list(SUPPORTED_MODELS.keys()),
                        help="Model to use for evaluation")
    parser.add_argument("--api-key", required=True, help="API key for the evaluator model")
    parser.add_argument("--input", required=True, help="Input JSON file with emails")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--prompt-style", required=True, choices=["naturalistic", "debiased"],
                        help="Prompt style: naturalistic or debiased")
    parser.add_argument("--checkpoint-interval", type=int, default=10,
                        help="Save checkpoint every N ratings")
    args = parser.parse_args()
    
    # Load emails
    with open(args.input, 'r') as f:
        emails = json.load(f)
    print(f"Loaded {len(emails)} emails from {args.input}")
    
    # Filter out emails with errors
    valid_emails = [e for e in emails if e.get("email_text") and not e.get("error")]
    print(f"Valid emails: {len(valid_emails)}")
    
    # Load checkpoint if exists
    completed = load_checkpoint(args.output)
    print(f"Loaded checkpoint: {len(completed)} ratings already completed")
    
    # Build task list
    tasks = [e for e in valid_emails if e["response_id"] not in completed]
    
    # Shuffle to avoid systematic effects
    random.shuffle(tasks)
    
    print(f"Remaining tasks: {len(tasks)}")
    print(f"Evaluator: {args.evaluator} ({SUPPORTED_MODELS[args.evaluator]['model_id']})")
    print(f"Prompt style: {args.prompt_style}")
    
    # Rate emails
    results = list(completed.values())
    
    for i, email_data in enumerate(tasks):
        print(f"[{i+1}/{len(tasks)}] {email_data['response_id']}...", end=" ")
        
        result = rate_email(
            evaluator=args.evaluator,
            api_key=args.api_key,
            email_data=email_data,
            prompt_style=args.prompt_style,
        )
        
        results.append(result)
        
        if result.get("error"):
            print(f"ERROR: {result['error']}")
        elif result.get("parse_error"):
            print("OK (parse warning)")
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
    errors = sum(1 for r in results if r.get("error"))
    parse_errors = sum(1 for r in results if r.get("parse_error") and not r.get("error"))
    print(f"\nComplete: {len(results)} ratings, {errors} errors, {parse_errors} parse warnings")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
