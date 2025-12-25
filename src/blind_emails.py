#!/usr/bin/env python3
"""
Create blinded versions of emails by removing persona names.

Usage:
    python blind_emails.py --input emails.json --output emails_blinded.json
"""

import argparse
import json
import re
from typing import List, Dict, Any


def blind_email(email_text: str, persona_name: str) -> str:
    """
    Replace persona name with [SENDER] in email text.
    
    Handles:
    - Full name (e.g., "Emily Chen")
    - First name only (e.g., "Emily")
    - Name in signature
    """
    parts = persona_name.split()
    first_name = parts[0]
    full_name = persona_name
    
    blinded = email_text
    
    # Replace full name first (to avoid partial replacements)
    if full_name:
        blinded = re.sub(re.escape(full_name), '[SENDER]', blinded, flags=re.IGNORECASE)
    
    # Replace first name (with word boundaries to avoid partial matches)
    if first_name:
        blinded = re.sub(r'\b' + re.escape(first_name) + r'\b', '[SENDER]', blinded, flags=re.IGNORECASE)
    
    return blinded


def blind_emails(emails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create blinded versions of all emails.
    """
    blinded = []
    
    for e in emails:
        if not e.get('email_text'):
            continue
        
        new_email = e.copy()
        new_email['email_text_original'] = e['email_text']
        new_email['email_text'] = blind_email(e['email_text'], e['persona_name'])
        new_email['blinded'] = True
        blinded.append(new_email)
    
    return blinded


def verify_blinding(emails: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Verify that blinding was successful.
    """
    issues = []
    
    for e in emails:
        text = e['email_text']
        name = e['persona_name']
        first_name = name.split()[0]
        
        # Check if name still appears
        if first_name.lower() in text.lower():
            issues.append({
                "response_id": e['response_id'],
                "issue": f"First name '{first_name}' still present",
            })
        
        if name.lower() in text.lower():
            issues.append({
                "response_id": e['response_id'],
                "issue": f"Full name '{name}' still present",
            })
    
    return {
        "total_emails": len(emails),
        "issues_found": len(issues),
        "issues": issues[:10],  # Only show first 10
    }


def main():
    parser = argparse.ArgumentParser(description="Blind emails by removing persona names")
    parser.add_argument("--input", required=True, help="Input JSON file with emails")
    parser.add_argument("--output", required=True, help="Output JSON file for blinded emails")
    parser.add_argument("--verify", action="store_true", help="Verify blinding was successful")
    args = parser.parse_args()
    
    # Load emails
    with open(args.input, 'r') as f:
        emails = json.load(f)
    print(f"Loaded {len(emails)} emails from {args.input}")
    
    # Blind emails
    blinded = blind_emails(emails)
    print(f"Blinded {len(blinded)} emails")
    
    # Verify
    if args.verify:
        verification = verify_blinding(blinded)
        print(f"\nVerification:")
        print(f"  Total: {verification['total_emails']}")
        print(f"  Issues: {verification['issues_found']}")
        if verification['issues']:
            print(f"  Sample issues:")
            for issue in verification['issues'][:5]:
                print(f"    - {issue['response_id']}: {issue['issue']}")
    
    # Save
    with open(args.output, 'w') as f:
        json.dump(blinded, f, indent=2)
    print(f"\nSaved to {args.output}")
    
    # Show example
    if blinded:
        print(f"\nExample (original):")
        print(f"  ...{blinded[0]['email_text_original'][-100:]}")
        print(f"\nExample (blinded):")
        print(f"  ...{blinded[0]['email_text'][-100:]}")


if __name__ == "__main__":
    main()
