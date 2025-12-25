#!/usr/bin/env python3
"""
Analyze gender bias in LLM-generated emails and ratings.

This script reproduces all tables from the paper. By default, it uses persona-level
aggregation (the approach used in the paper), but can also run email-level analysis
for comparison.

**Why persona-level?**
Each persona generates 3 emails. Treating these as independent observations inflates
the effective sample size (n=90 vs n=30) and underestimates standard errors, leading
to anti-conservative p-values. Aggregating to persona level respects the clustering
structure of the data.

Usage:
    # Paper approach (persona-level, recommended)
    python analyze_results.py --data-dir ../data/raw --output-dir ../data/processed
    
    # Email-level for comparison (NOT recommended, shown for transparency)
    python analyze_results.py --data-dir ../data/raw --output-dir ../data/processed --email-level
"""

import json
import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def load_emails(filepath: Path) -> List[dict]:
    """Load email JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_ratings(filepath: Path) -> List[dict]:
    """Load ratings JSON file, fixing parse errors where possible."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Fix parse errors by extracting from raw_response
    for r in data:
        if r.get('parse_error') and r.get('raw_response'):
            raw = r['raw_response']
            for field in ['likelihood_to_grant_raise', 'professionalism', 
                         'perceived_confidence', 'perceived_competence',
                         'likelihood_to_send_correction', 'perceived_reasonableness', 
                         'seems_entitled']:
                if field not in r:
                    match = re.search(rf'"{field}":\s*(\d)', raw)
                    if match:
                        r[field] = int(match.group(1))
    return data


def calc_persona_pattern(emails: List[dict], scenario: str, pattern: str, 
                         flags: int = re.IGNORECASE) -> Dict:
    """
    Calculate pattern frequency at persona level.
    
    Returns dict with F%, M%, p-value, Cohen's d
    """
    subset = [e for e in emails if e['scenario_id'] == scenario and e.get('email_text')]
    
    # Aggregate by persona
    f_props = defaultdict(list)
    m_props = defaultdict(list)
    
    for e in subset:
        has_pattern = 1 if re.search(pattern, e['email_text'], flags) else 0
        if e['gender'] == 'F':
            f_props[e['persona_id']].append(has_pattern)
        else:
            m_props[e['persona_id']].append(has_pattern)
    
    # Compute persona-level means
    f_means = [np.mean(v) for v in f_props.values()]
    m_means = [np.mean(v) for v in m_props.values()]
    
    f_mean = np.mean(f_means)
    m_mean = np.mean(m_means)
    
    # Mann-Whitney U test (non-parametric)
    _, p = stats.mannwhitneyu(f_means, m_means, alternative='two-sided')
    
    # Cohen's d
    pooled_std = np.sqrt((np.std(f_means, ddof=1)**2 + np.std(m_means, ddof=1)**2) / 2)
    d = (f_mean - m_mean) / pooled_std if pooled_std > 0 else 0
    
    return {
        'f_pct': f_mean * 100,
        'm_pct': m_mean * 100,
        'diff': (f_mean - m_mean) * 100,
        'p': p,
        'd': d,
        'n_f': len(f_means),
        'n_m': len(m_means)
    }


# =============================================================================
# EMAIL-LEVEL ANALYSIS (for comparison - NOT recommended)
# =============================================================================

def calc_email_pattern(emails: List[dict], scenario: str, pattern: str, 
                       flags: int = re.IGNORECASE) -> Dict:
    """
    Calculate pattern frequency at email level (treats each email as independent).
    
    WARNING: This approach inflates sample size and underestimates standard errors.
    Use calc_persona_pattern instead for correct inference.
    """
    subset = [e for e in emails if e['scenario_id'] == scenario and e.get('email_text')]
    
    f_emails = [e for e in subset if e['gender'] == 'F']
    m_emails = [e for e in subset if e['gender'] == 'M']
    
    f_count = sum(1 for e in f_emails if re.search(pattern, e['email_text'], flags))
    m_count = sum(1 for e in m_emails if re.search(pattern, e['email_text'], flags))
    
    f_pct = 100 * f_count / len(f_emails) if f_emails else 0
    m_pct = 100 * m_count / len(m_emails) if m_emails else 0
    
    # Fisher's exact test (email-level)
    _, p = stats.fisher_exact([
        [f_count, len(f_emails) - f_count],
        [m_count, len(m_emails) - m_count]
    ])
    
    # Cohen's h for proportions
    f_prop = f_count / len(f_emails) if f_emails else 0
    m_prop = m_count / len(m_emails) if m_emails else 0
    h = 2 * (np.arcsin(np.sqrt(f_prop)) - np.arcsin(np.sqrt(m_prop)))
    
    return {
        'f_pct': f_pct,
        'm_pct': m_pct,
        'diff': f_pct - m_pct,
        'p': p,
        'h': h,  # Cohen's h instead of d for proportions
        'n_f': len(f_emails),
        'n_m': len(m_emails)
    }


def calc_email_rating(ratings: List[dict], scenario: str, 
                      measure: str) -> Dict:
    """
    Calculate rating difference at email level (treats each rating as independent).
    
    WARNING: This approach inflates sample size and underestimates standard errors.
    Use calc_persona_rating instead for correct inference.
    """
    subset = [r for r in ratings if r['scenario_id'] == scenario]
    
    f_vals = [r[measure] for r in subset if r['gender'] == 'F' and r.get(measure) is not None]
    m_vals = [r[measure] for r in subset if r['gender'] == 'M' and r.get(measure) is not None]
    
    f_mean = np.mean(f_vals)
    m_mean = np.mean(m_vals)
    diff = f_mean - m_mean
    
    # Mann-Whitney U test (email-level)
    _, p = stats.mannwhitneyu(f_vals, m_vals, alternative='two-sided')
    
    # Cohen's d
    pooled_std = np.sqrt((np.std(f_vals, ddof=1)**2 + np.std(m_vals, ddof=1)**2) / 2)
    d = diff / pooled_std if pooled_std > 0 else 0
    
    return {
        'f_mean': f_mean,
        'm_mean': m_mean,
        'diff': diff,
        'p': p,
        'd': d,
        'n_f': len(f_vals),
        'n_m': len(m_vals)
    }


# =============================================================================
# PERSONA-LEVEL ANALYSIS (continued)
# =============================================================================


def calc_persona_rating(ratings: List[dict], scenario: str, 
                        measure: str) -> Dict:
    """
    Calculate rating difference at persona level.
    
    Returns dict with F mean, M mean, diff, p-value, Cohen's d
    """
    subset = [r for r in ratings if r['scenario_id'] == scenario]
    
    # Aggregate by persona
    f_vals = defaultdict(list)
    m_vals = defaultdict(list)
    
    for r in subset:
        val = r.get(measure)
        if val is not None:
            if r['gender'] == 'F':
                f_vals[r['persona_id']].append(val)
            else:
                m_vals[r['persona_id']].append(val)
    
    # Compute persona-level means
    f_means = [np.mean(v) for v in f_vals.values()]
    m_means = [np.mean(v) for v in m_vals.values()]
    
    f_mean = np.mean(f_means)
    m_mean = np.mean(m_means)
    diff = f_mean - m_mean
    
    # Mann-Whitney U test
    _, p = stats.mannwhitneyu(f_means, m_means, alternative='two-sided')
    
    # Cohen's d
    pooled_std = np.sqrt((np.std(f_means, ddof=1)**2 + np.std(m_means, ddof=1)**2) / 2)
    d = diff / pooled_std if pooled_std > 0 else 0
    
    return {
        'f_mean': f_mean,
        'm_mean': m_mean,
        'diff': diff,
        'p': p,
        'd': d,
        'n_f': len(f_means),
        'n_m': len(m_means)
    }


def benjamini_hochberg(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Apply Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    
    # Calculate BH threshold for each rank
    thresholds = [(i + 1) / n * alpha for i in range(n)]
    
    # Find largest k where p(k) <= threshold(k)
    significant = np.zeros(n, dtype=bool)
    max_k = -1
    for k in range(n):
        if sorted_p[k] <= thresholds[k]:
            max_k = k
    
    # All tests up to max_k are significant
    if max_k >= 0:
        significant[sorted_indices[:max_k + 1]] = True
    
    return significant.tolist()


def reproduce_table2(gpt_emails: List[dict], gemini_emails: List[dict]) -> pd.DataFrame:
    """Reproduce Table 2: Generation patterns (FDR-corrected)."""
    
    patterns = [
        ('GPT-5.2', 'S01', 'given_my', r'given my', re.IGNORECASE),
        ('GPT-5.2', 'S01', 'i_believe', r'\bi believe\b', re.IGNORECASE),
        ('GPT-5.2', 'S02', 'wanted_to', r'wanted to', re.IGNORECASE),
        ('GPT-5.2', 'S02', 'full_name_sig', r'\n[A-Z][a-z]+ [A-Z][a-z]+\s*$', 0),
        ('GPT-5.2', 'S02', 'clarify', r'\bclarify\b', re.IGNORECASE),
        ('GPT-5.2', 'S02', 'follow_up', r'follow.?up', re.IGNORECASE),
        ('Gemini', 'S01', 'given_my', r'given my', re.IGNORECASE),
        ('Gemini', 'S01', 'i_believe', r'\bi believe\b', re.IGNORECASE),
        ('Gemini', 'S02', 'wanted_to', r'wanted to', re.IGNORECASE),
        ('Gemini', 'S02', 'full_name_sig', r'\n[A-Z][a-z]+ [A-Z][a-z]+\s*$', 0),
        ('Gemini', 'S02', 'clarify', r'\bclarify\b', re.IGNORECASE),
        ('Gemini', 'S02', 'follow_up', r'follow.?up', re.IGNORECASE),
    ]
    
    results = []
    for model, scenario, name, pattern, flags in patterns:
        emails = gpt_emails if model == 'GPT-5.2' else gemini_emails
        res = calc_persona_pattern(emails, scenario, pattern, flags)
        results.append({
            'model': model,
            'scenario': scenario,
            'pattern': name,
            'f_pct': res['f_pct'],
            'm_pct': res['m_pct'],
            'diff': res['diff'],
            'p': res['p'],
            'd': res['d']
        })
    
    df = pd.DataFrame(results)
    
    # Apply BH correction
    df['significant'] = benjamini_hochberg(df['p'].tolist())
    
    return df


def reproduce_table3(ratings_dict: Dict[str, List[dict]]) -> pd.DataFrame:
    """Reproduce Table 3: S01 Evaluation (no significant differences)."""
    
    results = []
    for key, label in [
        ('gpt52_gemini_nat', 'GPT-5.2 → Gemini 2.0 Naturalistic'),
        ('gpt52_gemini_deb', 'GPT-5.2 → Gemini 2.0 Debiased'),
        ('gemini_gpt52_nat', 'Gemini 2.0 → GPT-5.2 Naturalistic'),
        ('gemini_gpt52_deb', 'Gemini 2.0 → GPT-5.2 Debiased'),
    ]:
        ratings = ratings_dict[key]
        res = calc_persona_rating(ratings, 'S01', 'likelihood_to_grant_raise')
        results.append({
            'setting': label,
            'f_mean': res['f_mean'],
            'm_mean': res['m_mean'],
            'diff': res['diff'],
            'p': res['p']
        })
    
    return pd.DataFrame(results)


def reproduce_table4(ratings_dict: Dict[str, List[dict]]) -> pd.DataFrame:
    """Reproduce Table 4: S02 Evaluation."""
    
    results = []
    for key, label in [
        ('gpt52_gemini_nat', 'GPT-5.2 → Gemini 2.0 Naturalistic'),
        ('gpt52_gemini_deb', 'GPT-5.2 → Gemini 2.0 Debiased'),
        ('gpt52_gemini_blind', 'GPT-5.2 → Gemini 2.0 Blinded'),
        ('gemini_gpt52_nat', 'Gemini 2.0 → GPT-5.2 Naturalistic'),
        ('gemini_gpt52_deb', 'Gemini 2.0 → GPT-5.2 Debiased'),
        ('gemini_gpt52_blind', 'Gemini 2.0 → GPT-5.2 Blinded'),
    ]:
        ratings = ratings_dict[key]
        res = calc_persona_rating(ratings, 'S02', 'likelihood_to_send_correction')
        results.append({
            'setting': label,
            'diff': res['diff'],
            'p': res['p'],
            'd': res['d']
        })
    
    return pd.DataFrame(results)


def reproduce_table5(ratings_dict: Dict[str, List[dict]]) -> pd.DataFrame:
    """Reproduce Table 5: Blinded decomposition."""
    
    results = []
    
    for evaluator, gen_key, eval_key_base in [
        ('Gemini 2.0', 'gpt52', 'gpt52_gemini'),
        ('GPT-5.2', 'gemini', 'gemini_gpt52'),
    ]:
        nat = calc_persona_rating(ratings_dict[f'{eval_key_base}_nat'], 'S02', 'likelihood_to_send_correction')
        blind = calc_persona_rating(ratings_dict[f'{eval_key_base}_blind'], 'S02', 'likelihood_to_send_correction')
        deb = calc_persona_rating(ratings_dict[f'{eval_key_base}_deb'], 'S02', 'likelihood_to_send_correction')
        
        name_component = nat['diff'] - blind['diff']
        style_component = blind['diff']
        
        results.append({
            'evaluator': evaluator,
            'unblinded': nat['diff'],
            'unblinded_d': nat['d'],
            'blinded': blind['diff'],
            'blinded_d': blind['d'],
            'debiased': deb['diff'],
            'name_component': name_component,
            'style_component': style_component,
            'interpretation': 'Pure style' if abs(name_component) < 0.05 else 'Name + Style'
        })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Analyze gender bias results')
    parser.add_argument('--data-dir', type=str, default='../data/raw',
                        help='Directory containing raw data files')
    parser.add_argument('--output-dir', type=str, default='../data/processed',
                        help='Directory for output CSVs')
    parser.add_argument('--email-level', action='store_true',
                        help='Use email-level analysis (NOT recommended, inflates sample size)')
    parser.add_argument('--compare', action='store_true',
                        help='Show both persona-level and email-level for comparison')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    
    # Load emails
    gpt_emails = load_emails(data_dir / 'emails_gpt52.json')
    gemini_emails = load_emails(data_dir / 'emails_gemini.json')
    
    # Load ratings
    ratings = {
        'gpt52_gemini_nat': load_ratings(data_dir / 'ratings_gemini_naturalistic.json'),
        'gpt52_gemini_deb': load_ratings(data_dir / 'ratings_gemini_debiased.json'),
        'gpt52_gemini_blind': load_ratings(data_dir / 'ratings_gemini_blinded.json'),
        'gemini_gpt52_nat': load_ratings(data_dir / 'ratings_gpt52_naturalistic.json'),
        'gemini_gpt52_deb': load_ratings(data_dir / 'ratings_gpt52_debiased.json'),
        'gemini_gpt52_blind': load_ratings(data_dir / 'ratings_gpt52_blinded.json'),
    }
    
    print(f"Loaded {len(gpt_emails)} GPT-5.2 emails, {len(gemini_emails)} Gemini emails")
    print(f"Loaded {sum(len(v) for v in ratings.values())} total ratings")
    
    if args.compare:
        run_comparison(gpt_emails, gemini_emails, ratings, output_dir)
    elif args.email_level:
        print("\n⚠️  WARNING: Email-level analysis treats each email as independent.")
        print("   This inflates sample size (n=90 vs n=30) and underestimates p-values.")
        print("   Use --compare to see the difference, or omit --email-level for correct analysis.\n")
        run_email_level(gpt_emails, gemini_emails, ratings, output_dir)
    else:
        run_persona_level(gpt_emails, gemini_emails, ratings, output_dir)


def run_persona_level(gpt_emails, gemini_emails, ratings, output_dir):
    """Run persona-level analysis (paper approach)."""
    print("\n" + "="*70)
    print("PERSONA-LEVEL ANALYSIS (Paper approach, n=30 per group)")
    print("="*70)
    
    print("\n" + "-"*70)
    print("TABLE 2: Generation Patterns (FDR-corrected)")
    print("-"*70)
    table2 = reproduce_table2(gpt_emails, gemini_emails)
    sig_only = table2[table2['significant']]
    print(sig_only.to_string(index=False))
    table2.to_csv(output_dir / 'table2_generation_patterns.csv', index=False)
    
    print("\n" + "-"*70)
    print("TABLE 3: S01 Evaluation")
    print("-"*70)
    table3 = reproduce_table3(ratings)
    print(table3.to_string(index=False))
    table3.to_csv(output_dir / 'table3_s01_evaluation.csv', index=False)
    
    print("\n" + "-"*70)
    print("TABLE 4: S02 Evaluation")
    print("-"*70)
    table4 = reproduce_table4(ratings)
    print(table4.to_string(index=False))
    table4.to_csv(output_dir / 'table4_s02_evaluation.csv', index=False)
    
    print("\n" + "-"*70)
    print("TABLE 5: Blinded Decomposition")
    print("-"*70)
    table5 = reproduce_table5(ratings)
    print(table5.to_string(index=False))
    table5.to_csv(output_dir / 'table5_decomposition.csv', index=False)
    
    print(f"\nResults saved to {output_dir}/")


def run_email_level(gpt_emails, gemini_emails, ratings, output_dir):
    """Run email-level analysis (NOT recommended)."""
    print("\n" + "="*70)
    print("EMAIL-LEVEL ANALYSIS (n=90 per group - inflated sample size)")
    print("="*70)
    
    patterns = [
        ('GPT-5.2', 'S02', 'wanted_to', r'wanted to', re.IGNORECASE),
        ('GPT-5.2', 'S02', 'full_name_sig', r'\n[A-Z][a-z]+ [A-Z][a-z]+\s*$', 0),
        ('GPT-5.2', 'S02', 'clarify', r'\bclarify\b', re.IGNORECASE),
        ('Gemini', 'S01', 'i_believe', r'\bi believe\b', re.IGNORECASE),
        ('Gemini', 'S02', 'follow_up', r'follow.?up', re.IGNORECASE),
    ]
    
    print("\nGeneration patterns (email-level):")
    print(f"{'Model':<10} {'Pattern':<15} {'F%':>8} {'M%':>8} {'p':>10} {'n_F':>6} {'n_M':>6}")
    print("-"*70)
    for model, scenario, name, pattern, flags in patterns:
        emails = gpt_emails if model == 'GPT-5.2' else gemini_emails
        res = calc_email_pattern(emails, scenario, pattern, flags)
        print(f"{model:<10} {name:<15} {res['f_pct']:>7.1f}% {res['m_pct']:>7.1f}% {res['p']:>10.6f} {res['n_f']:>6} {res['n_m']:>6}")
    
    print("\nS02 Evaluation (email-level):")
    print(f"{'Setting':<45} {'Diff':>8} {'p':>12} {'n_F':>6} {'n_M':>6}")
    print("-"*80)
    for key, label in [
        ('gemini_gpt52_nat', 'Gemini 2.0 → GPT-5.2 Naturalistic'),
        ('gemini_gpt52_deb', 'Gemini 2.0 → GPT-5.2 Debiased'),
        ('gemini_gpt52_blind', 'Gemini 2.0 → GPT-5.2 Blinded'),
    ]:
        res = calc_email_rating(ratings[key], 'S02', 'likelihood_to_send_correction')
        print(f"{label:<45} {res['diff']:>+7.2f} {res['p']:>12.6f} {res['n_f']:>6} {res['n_m']:>6}")


def run_comparison(gpt_emails, gemini_emails, ratings, output_dir):
    """Compare persona-level vs email-level analysis."""
    print("\n" + "="*70)
    print("COMPARISON: Persona-level vs Email-level Analysis")
    print("="*70)
    print("\nThis demonstrates why persona-level aggregation is necessary.")
    print("Email-level treats 3 responses per persona as independent, inflating n.\n")
    
    patterns = [
        ('GPT-5.2', 'S02', 'wanted_to', r'wanted to', re.IGNORECASE),
        ('GPT-5.2', 'S02', 'full_name_sig', r'\n[A-Z][a-z]+ [A-Z][a-z]+\s*$', 0),
        ('Gemini', 'S02', 'follow_up', r'follow.?up', re.IGNORECASE),
    ]
    
    print("GENERATION PATTERNS:")
    print(f"{'Pattern':<20} {'Level':<10} {'n':>6} {'p-value':>12} {'Note':<20}")
    print("-"*75)
    
    for model, scenario, name, pattern, flags in patterns:
        emails = gpt_emails if model == 'GPT-5.2' else gemini_emails
        
        persona = calc_persona_pattern(emails, scenario, pattern, flags)
        email = calc_email_pattern(emails, scenario, pattern, flags)
        
        print(f"{model} {name:<13} {'Persona':<10} {persona['n_f']:>6} {persona['p']:>12.6f} {'← CORRECT':<20}")
        print(f"{'':<20} {'Email':<10} {email['n_f']:>6} {email['p']:>12.6f} {'(inflated n)':<20}")
        print()
    
    print("\nEVALUATION (GPT-5.2 evaluator, S02):")
    print(f"{'Condition':<15} {'Level':<10} {'n':>6} {'Diff':>8} {'p-value':>12}")
    print("-"*60)
    
    for key, label in [
        ('gemini_gpt52_nat', 'Naturalistic'),
        ('gemini_gpt52_blind', 'Blinded'),
        ('gemini_gpt52_deb', 'Debiased'),
    ]:
        persona = calc_persona_rating(ratings[key], 'S02', 'likelihood_to_send_correction')
        email = calc_email_rating(ratings[key], 'S02', 'likelihood_to_send_correction')
        
        print(f"{label:<15} {'Persona':<10} {persona['n_f']:>6} {persona['diff']:>+7.2f} {persona['p']:>12.6f}")
        print(f"{'':<15} {'Email':<10} {email['n_f']:>6} {email['diff']:>+7.2f} {email['p']:>12.6f}")
        print()
    
    print("KEY INSIGHT: P-values are similar because effects are large,")
    print("but email-level would be anti-conservative for smaller effects.")
    print("\nThe paper uses persona-level (n=30) to correctly account for clustering.")


if __name__ == '__main__':
    main()
