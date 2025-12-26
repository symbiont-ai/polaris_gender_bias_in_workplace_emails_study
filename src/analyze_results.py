#!/usr/bin/env python3
"""
Analyze gender bias in LLM-generated emails and ratings.

This script reproduces ALL tables from the paper (Tables 2-8) using paired statistics.
Unit of analysis: matched persona pairs (n=30 pairs).

Statistical approach:
- Within-pair differences (Female - Male) computed for each pair
- Wilcoxon signed-rank test for significance
- 95% confidence intervals for all effects
- Benjamini-Hochberg FDR correction for multiple comparisons (Table 2 only)

Usage:
    python analyze_results.py --data-dir ../data/raw --output-dir ../data/processed

Tables reproduced:
    Table 2: Generation patterns (significant only, with q-values)
    Table 3: S01 evaluation (primary outcome)
    Table 4: S02 evaluation (primary outcome)
    Table 5: Bias decomposition
    Table 6: All 24 generation patterns (Appendix A)
    Table 7: S01 secondary outcomes (Appendix B)
    Table 8: S02 secondary outcomes (Appendix B)
"""

import json
import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from collections import defaultdict


def load_emails(filepath):
    """Load email JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_ratings(filepath):
    """Load ratings JSON file, fixing parse errors where possible."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
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


def paired_pattern_analysis(emails, scenario, pattern, flags=re.IGNORECASE):
    """Compute within-pair differences for a generation pattern."""
    subset = [e for e in emails if e['scenario_id'] == scenario and e.get('email_text')]
    persona_props = defaultdict(list)
    for e in subset:
        has = 1 if re.search(pattern, e['email_text'], flags) else 0
        persona_props[e['persona_id']].append(has)
    
    persona_means = {pid: np.mean(vals) for pid, vals in persona_props.items()}
    
    f_means = [persona_means.get(f"F{i:02d}", np.nan) for i in range(1, 31)]
    m_means = [persona_means.get(f"M{i:02d}", np.nan) for i in range(1, 31)]
    f_means = [x for x in f_means if not np.isnan(x)]
    m_means = [x for x in m_means if not np.isnan(x)]
    
    pair_diffs = []
    for i in range(1, 31):
        f_pid, m_pid = f"F{i:02d}", f"M{i:02d}"
        if f_pid in persona_means and m_pid in persona_means:
            pair_diffs.append(persona_means[f_pid] - persona_means[m_pid])
    pair_diffs = np.array(pair_diffs)
    
    if np.all(pair_diffs == 0) or len(set(pair_diffs)) < 2:
        return {
            'f_pct': np.mean(f_means) * 100 if f_means else 0,
            'm_pct': np.mean(m_means) * 100 if m_means else 0,
            'diff': 0, 'ci_low': 0, 'ci_high': 0,
            'p': np.nan, 'd': 0, 'n': len(pair_diffs), 'testable': False
        }
    
    nonzero = pair_diffs[pair_diffs != 0]
    if len(nonzero) >= 2:
        # Use method='auto' for compatibility across scipy versions
        # With ties and n>13, this uses the normal approximation
        _, p = stats.wilcoxon(nonzero, alternative='two-sided', method='auto')
    else:
        p = 1.0
    
    mean_diff = np.mean(pair_diffs)
    std_diff = np.std(pair_diffs, ddof=1)
    d = mean_diff / std_diff if std_diff > 0 else 0
    se = std_diff / np.sqrt(len(pair_diffs))
    ci_low = mean_diff - 1.96 * se
    ci_high = mean_diff + 1.96 * se
    
    return {
        'f_pct': np.mean(f_means) * 100,
        'm_pct': np.mean(m_means) * 100,
        'diff': mean_diff * 100,
        'ci_low': ci_low * 100,
        'ci_high': ci_high * 100,
        'p': p, 'd': d, 'n': len(pair_diffs), 'testable': True
    }


def paired_rating_analysis(data, scenario, measure):
    """Compute within-pair differences for evaluation ratings."""
    subset = [r for r in data if r['scenario_id'] == scenario]
    persona_vals = defaultdict(list)
    for r in subset:
        val = r.get(measure)
        if val is not None:
            persona_vals[r['persona_id']].append(val)
    
    persona_means = {pid: np.mean(vals) for pid, vals in persona_vals.items()}
    
    pair_diffs = []
    for i in range(1, 31):
        f_pid, m_pid = f"F{i:02d}", f"M{i:02d}"
        if f_pid in persona_means and m_pid in persona_means:
            pair_diffs.append(persona_means[f_pid] - persona_means[m_pid])
    pair_diffs = np.array(pair_diffs)
    
    if len(pair_diffs) == 0:
        return None
    
    nonzero = pair_diffs[pair_diffs != 0]
    if len(nonzero) >= 2:
        # Use method='auto' for compatibility across scipy versions
        _, p = stats.wilcoxon(nonzero, alternative='two-sided', method='auto')
    else:
        p = 1.0
    
    mean_diff = np.mean(pair_diffs)
    std_diff = np.std(pair_diffs, ddof=1)
    d = mean_diff / std_diff if std_diff > 0 else 0
    se = std_diff / np.sqrt(len(pair_diffs))
    ci_low = mean_diff - 1.96 * se
    ci_high = mean_diff + 1.96 * se
    
    return {
        'diff': mean_diff,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'p': p, 'd': d, 'n': len(pair_diffs)
    }


def apply_bh_correction(results, alpha=0.05):
    """Apply Benjamini-Hochberg FDR correction."""
    testable = [r for r in results if r.get('testable', True) and not np.isnan(r.get('p', np.nan))]
    testable_sorted = sorted(testable, key=lambda x: x['p'])
    n = len(testable_sorted)
    
    for i, r in enumerate(testable_sorted):
        rank = i + 1
        r['bh_thresh'] = rank / n * alpha
        r['q'] = min(r['p'] * n / rank, 1.0)
        r['significant'] = r['p'] <= r['bh_thresh']
    
    for r in results:
        if not r.get('testable', True) or np.isnan(r.get('p', np.nan)):
            r['significant'] = False
            r['bh_thresh'] = np.nan
            r['q'] = np.nan
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze gender bias in LLM emails')
    parser.add_argument('--data-dir', type=str, default='../data/raw')
    parser.add_argument('--output-dir', type=str, default='../data/processed')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("REPRODUCING ALL PAPER TABLES")
    print("="*70)
    print("\nLoading data...")
    
    gpt_emails = load_emails(data_dir / 'emails_gpt52.json')
    gemini_emails = load_emails(data_dir / 'emails_gemini.json')
    
    ratings = {
        'gemini_nat': load_ratings(data_dir / 'ratings_gemini_naturalistic.json'),
        'gemini_deb': load_ratings(data_dir / 'ratings_gemini_debiased.json'),
        'gemini_blind': load_ratings(data_dir / 'ratings_gemini_blinded.json'),
        'gpt52_nat': load_ratings(data_dir / 'ratings_gpt52_naturalistic.json'),
        'gpt52_deb': load_ratings(data_dir / 'ratings_gpt52_debiased.json'),
        'gpt52_blind': load_ratings(data_dir / 'ratings_gpt52_blinded.json'),
    }
    
    print(f"Loaded {len(gpt_emails)} GPT-5.2 emails, {len(gemini_emails)} Gemini 2.0 emails")
    print(f"Loaded {sum(len(v) for v in ratings.values())} total ratings")
    
    patterns = [
        ('i_believe', r'\bi believe\b', re.IGNORECASE),
        ('given_my', r'given my', re.IGNORECASE),
        ('wanted_to', r'wanted to', re.IGNORECASE),
        ('follow_up', r'follow.?up', re.IGNORECASE),
        ('clarify', r'\bclarify\b', re.IGNORECASE),
        ('full_name_sig', r'\n[A-Z][a-z]+ [A-Z][a-z]+\s*$', 0),
    ]
    
    # =========================================================================
    # TABLE 2 & 6: Generation Patterns
    # =========================================================================
    print("\n" + "="*70)
    print("TABLE 2: GENERATION PATTERNS (Significant, with q-values)")
    print("TABLE 6: ALL 24 GENERATION PATTERNS (Appendix A)")
    print("="*70)
    
    all_patterns = []
    for scenario in ['S01', 'S02']:
        for model_name, emails in [('GPT-5.2', gpt_emails), ('Gemini 2.0', gemini_emails)]:
            for pattern_name, pattern, flags in patterns:
                res = paired_pattern_analysis(emails, scenario, pattern, flags)
                all_patterns.append({
                    'scenario': scenario, 'model': model_name, 'pattern': pattern_name, **res
                })
    
    all_patterns = apply_bh_correction(all_patterns)
    pd.DataFrame(all_patterns).to_csv(output_dir / 'table6_all_generation_patterns.csv', index=False)
    
    # Table 2: significant only
    sig_patterns = [p for p in all_patterns if p['significant']]
    pd.DataFrame(sig_patterns).to_csv(output_dir / 'table2_significant_patterns.csv', index=False)
    
    print(f"\nTable 2 - Significant: {len(sig_patterns)} / {sum(1 for p in all_patterns if p['testable'])} testable")
    for p in sorted(sig_patterns, key=lambda x: (x['scenario'], x['model'])):
        print(f"  {p['scenario']} {p['model']:<12} {p['pattern']:<15} p={p['p']:.3f} q={p['q']:.3f}")
    
    # =========================================================================
    # TABLE 3: S01 Evaluation
    # =========================================================================
    print("\n" + "="*70)
    print("TABLE 3: S01 EVALUATION (Primary outcome)")
    print("="*70)
    
    s01_results = []
    for setting, condition, key in [
        ('GPT-5.2 → Gemini 2.0', 'Naturalistic', 'gemini_nat'),
        ('GPT-5.2 → Gemini 2.0', 'Debiased', 'gemini_deb'),
        ('GPT-5.2 → Gemini 2.0', 'Blinded', 'gemini_blind'),
        ('Gemini 2.0 → GPT-5.2', 'Naturalistic', 'gpt52_nat'),
        ('Gemini 2.0 → GPT-5.2', 'Debiased', 'gpt52_deb'),
        ('Gemini 2.0 → GPT-5.2', 'Blinded', 'gpt52_blind'),
    ]:
        res = paired_rating_analysis(ratings[key], 'S01', 'likelihood_to_grant_raise')
        if res:
            s01_results.append({'setting': setting, 'condition': condition, **res})
            print(f"  {setting:<25} {condition:<12} {res['diff']:+.2f} [{res['ci_low']:+.2f}, {res['ci_high']:+.2f}]")
    
    pd.DataFrame(s01_results).to_csv(output_dir / 'table3_s01_evaluation.csv', index=False)
    
    # =========================================================================
    # TABLE 4: S02 Evaluation
    # =========================================================================
    print("\n" + "="*70)
    print("TABLE 4: S02 EVALUATION (Primary outcome)")
    print("="*70)
    
    s02_results = []
    for setting, condition, key in [
        ('GPT-5.2 → Gemini 2.0', 'Naturalistic', 'gemini_nat'),
        ('GPT-5.2 → Gemini 2.0', 'Debiased', 'gemini_deb'),
        ('GPT-5.2 → Gemini 2.0', 'Blinded', 'gemini_blind'),
        ('Gemini 2.0 → GPT-5.2', 'Naturalistic', 'gpt52_nat'),
        ('Gemini 2.0 → GPT-5.2', 'Debiased', 'gpt52_deb'),
        ('Gemini 2.0 → GPT-5.2', 'Blinded', 'gpt52_blind'),
    ]:
        res = paired_rating_analysis(ratings[key], 'S02', 'likelihood_to_send_correction')
        if res:
            s02_results.append({'setting': setting, 'condition': condition, 'key': key, **res})
            p_str = f"{res['p']:.3f}" if res['p'] >= 0.001 else "<.001"
            print(f"  {setting:<25} {condition:<12} {res['diff']:+.2f} [{res['ci_low']:+.2f}, {res['ci_high']:+.2f}] p={p_str} d={res['d']:+.2f}")
    
    pd.DataFrame(s02_results).to_csv(output_dir / 'table4_s02_evaluation.csv', index=False)
    
    # =========================================================================
    # TABLE 5: Decomposition
    # =========================================================================
    print("\n" + "="*70)
    print("TABLE 5: BIAS DECOMPOSITION")
    print("="*70)
    
    s02_by_key = {r['key']: r for r in s02_results}
    decomp = []
    
    for evaluator, nat_key, blind_key, deb_key in [
        ('Gemini 2.0', 'gemini_nat', 'gemini_blind', 'gemini_deb'),
        ('GPT-5.2', 'gpt52_nat', 'gpt52_blind', 'gpt52_deb'),
    ]:
        nat = s02_by_key[nat_key]
        blind = s02_by_key[blind_key]
        deb = s02_by_key[deb_key]
        
        decomp.append({
            'evaluator': evaluator,
            'unblinded': nat['diff'],
            'unblinded_d': nat['d'],
            'blinded': blind['diff'],
            'blinded_d': blind['d'],
            'debiased': deb['diff'],
            'name_component': nat['diff'] - blind['diff'],
            'style_component': blind['diff'],
        })
        
        print(f"\n{evaluator} evaluator:")
        print(f"  Unblinded: {nat['diff']:+.2f} (d={nat['d']:.2f})")
        print(f"  Blinded:   {blind['diff']:+.2f} (d={blind['d']:.2f})")
        print(f"  Debiased:  {deb['diff']:+.2f}")
        print(f"  → Name component:  {nat['diff'] - blind['diff']:+.2f}")
        print(f"  → Style component: {blind['diff']:+.2f}")
    
    pd.DataFrame(decomp).to_csv(output_dir / 'table5_decomposition.csv', index=False)
    
    # =========================================================================
    # TABLE 7: S01 Secondary Outcomes (Appendix B)
    # =========================================================================
    print("\n" + "="*70)
    print("TABLE 7: S01 SECONDARY OUTCOMES (Appendix B, exploratory)")
    print("="*70)
    
    s01_secondary = []
    s01_measures = ['professionalism', 'perceived_confidence', 'perceived_competence']
    
    for setting, condition, key in [
        ('GPT-5.2 → Gemini 2.0', 'Nat', 'gemini_nat'),
        ('GPT-5.2 → Gemini 2.0', 'Deb', 'gemini_deb'),
        ('GPT-5.2 → Gemini 2.0', 'Blind', 'gemini_blind'),
        ('Gemini 2.0 → GPT-5.2', 'Nat', 'gpt52_nat'),
        ('Gemini 2.0 → GPT-5.2', 'Deb', 'gpt52_deb'),
        ('Gemini 2.0 → GPT-5.2', 'Blind', 'gpt52_blind'),
    ]:
        for measure in s01_measures:
            res = paired_rating_analysis(ratings[key], 'S01', measure)
            if res:
                s01_secondary.append({
                    'setting': setting, 'condition': condition, 
                    'measure': measure, **res
                })
    
    pd.DataFrame(s01_secondary).to_csv(output_dir / 'table7_s01_secondary.csv', index=False)
    print(f"  Saved {len(s01_secondary)} rows")
    
    # =========================================================================
    # TABLE 8: S02 Secondary Outcomes (Appendix B)
    # =========================================================================
    print("\n" + "="*70)
    print("TABLE 8: S02 SECONDARY OUTCOMES (Appendix B, exploratory)")
    print("="*70)
    
    s02_secondary = []
    s02_measures = ['professionalism', 'perceived_reasonableness', 'seems_entitled']
    
    for setting, condition, key in [
        ('GPT-5.2 → Gemini 2.0', 'Nat', 'gemini_nat'),
        ('GPT-5.2 → Gemini 2.0', 'Deb', 'gemini_deb'),
        ('GPT-5.2 → Gemini 2.0', 'Blind', 'gemini_blind'),
        ('Gemini 2.0 → GPT-5.2', 'Nat', 'gpt52_nat'),
        ('Gemini 2.0 → GPT-5.2', 'Deb', 'gpt52_deb'),
        ('Gemini 2.0 → GPT-5.2', 'Blind', 'gpt52_blind'),
    ]:
        for measure in s02_measures:
            res = paired_rating_analysis(ratings[key], 'S02', measure)
            if res:
                s02_secondary.append({
                    'setting': setting, 'condition': condition,
                    'measure': measure, **res
                })
    
    pd.DataFrame(s02_secondary).to_csv(output_dir / 'table8_s02_secondary.csv', index=False)
    print(f"  Saved {len(s02_secondary)} rows")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("OUTPUT FILES")
    print("="*70)
    print(f"""
Files saved to {output_dir}/:
  table2_significant_patterns.csv  - Table 2 (6 significant patterns)
  table3_s01_evaluation.csv        - Table 3 (S01 primary outcome)
  table4_s02_evaluation.csv        - Table 4 (S02 primary outcome)
  table5_decomposition.csv         - Table 5 (bias decomposition)
  table6_all_generation_patterns.csv - Table 6/Appendix A (all 24 patterns)
  table7_s01_secondary.csv         - Table 7/Appendix B (S01 secondary)
  table8_s02_secondary.csv         - Table 8/Appendix B (S02 secondary)
""")


if __name__ == '__main__':
    main()
