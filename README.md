# Disentangling Generation and LLM-Judge Effects in Workplace Emails: Gender-Coded Differences Across Models

This repository contains the code, data, and paper for our study on gender-coded differences in LLM-generated workplace emails and their evaluation by other LLMs.

## Models and Data Collection

- **GPT-5.2**: `gpt-5.2-2025-12-01` (OpenAI API)
- **Gemini 2.0 Flash**: `gemini-2.0-flash-exp` (Google API)
- **Collection dates**: December 15–20, 2025
- **Parameters**: temperature=1.0, top_p=1.0, no system prompt

## Key Findings

1. **Context-dependent effects**: Strong effects emerged in credit attribution (S02); in salary negotiation (S01) we observe no evaluation bias and only one generation-style difference (Gemini: "I believe")

2. **Different models encode different stereotypes**: 
   - GPT-5.2 generates female emails with softer framing ("wanted to": +15.6 pp, "clarify": +16.7 pp) and less formal signatures (-27.8 pp)
   - Gemini 2.0 generates female emails with collaborative framing ("follow-up": +25.6 pp)

3. **Evaluation bias decomposes into name-based and style-based components**: 
   - GPT-5.2's bias (+0.61) = name bias (+0.17) + style preference (+0.44)
   - Gemini 2.0's bias (+0.28) is entirely style-based

4. **Debiasing prompts have asymmetric effects**: "Be objective" eliminates GPT-5.2's bias but has no effect on Gemini 2.0

## Repository Structure

```
gender-bias-llm/
├── paper/
│   ├── gender_bias_llm.pdf      # The paper
│   └── gender_bias_llm.tex      # LaTeX source
├── src/
│   ├── config.py                # Prompts, personas, scenarios
│   ├── generate_emails.py       # Email generation script
│   ├── rate_emails.py           # Rating script
│   ├── blind_emails.py          # Blinding script
│   └── analyze_results.py       # Statistical analysis (paired)
├── data/
│   ├── raw/                     # Original JSON files
│   │   ├── emails_gpt52.json    # 360 GPT-5.2 generated emails
│   │   ├── emails_gemini.json   # 360 Gemini 2.0 generated emails
│   │   └── ratings_*.json       # All ratings (8 files)
│   └── processed/               # Analysis outputs (CSVs)
├── notebooks/
│   └── reproduce_tables.ipynb   # Jupyter notebook reproducing all tables
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick Start

### Reproduce All Tables (2-8)

```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis script
cd src
python analyze_results.py --data-dir ../data/raw --output-dir ../data/processed
```

This generates CSV files for all paper tables:
- `table2_significant_patterns.csv` - Table 2 (6 significant generation patterns)
- `table3_s01_evaluation.csv` - Table 3 (S01 primary outcome)
- `table4_s02_evaluation.csv` - Table 4 (S02 primary outcome)  
- `table5_decomposition.csv` - Table 5 (bias decomposition)
- `table6_all_generation_patterns.csv` - Table 6/Appendix A (all 24 patterns)
- `table7_s01_secondary.csv` - Table 7/Appendix B (S01 secondary, exploratory)
- `table8_s02_secondary.csv` - Table 8/Appendix B (S02 secondary, exploratory)

### Reproducibility Notes

- **Python**: Requires Python ≥3.11
- **scipy**: Pinned to `scipy==1.16.3` for exact reproducibility
- **Statistical tests**: Wilcoxon signed-rank with `method="auto"`, which in scipy 1.16.3 uses the asymptotic (normal approximation) method for our sample sizes (n>13 with ties). Other scipy versions may produce slightly different p-values.
- **Random seed**: Not applicable (no stochastic elements in analysis)

### Statistical Approach

We use **paired statistics** because personas are matched pairs (e.g., Emily Chen / Michael Chen):
- Within-pair differences (Female - Male) computed for each of 30 pairs
- Wilcoxon signed-rank tests for significance
- 95% confidence intervals for all effects
- Benjamini-Hochberg FDR correction (α = 0.05) for multiple comparisons

This approach is more powerful than unpaired tests and directly matches the experimental design.

## Data Format

### Emails (JSON)
```json
{
  "persona_id": "F01",
  "persona_name": "Emily Chen",
  "gender": "F",
  "scenario_id": "S01",
  "email_text": "Subject: Discussion About Compensation...",
  "generator_model": "gpt-5.2"
}
```

### Ratings (JSON)
```json
{
  "persona_id": "F01",
  "gender": "F",
  "scenario_id": "S02",
  "likelihood_to_send_correction": 4,
  "professionalism": 5,
  "perceived_reasonableness": 4
}
```

## Key Statistics from Paper

### Generation Patterns (Table 2, 6 survive BH-FDR)

| Model | Pattern | F% | M% | Diff [95% CI] | p | d |
|-------|---------|----|----|---------------|---|---|
| Gemini 2.0 | "I believe" (S01) | 10.0% | 26.7% | -16.7 [-28, -5] | .010 | -0.51 |
| GPT-5.2 | "clarify" (S02) | 97.8% | 81.1% | +16.7 [+9, +25] | .001 | +0.73 |
| Gemini 2.0 | "follow-up" (S02) | 67.8% | 42.2% | +25.6 [+14, +37] | .001 | +0.79 |
| GPT-5.2 | "wanted to" (S02) | 95.6% | 80.0% | +15.6 [+7, +24] | .002 | +0.68 |
| GPT-5.2 | full name sig (S02) | 47.8% | 75.6% | -27.8 [-43, -13] | .002 | -0.68 |
| Gemini 2.0 | "clarify" (S02) | 41.1% | 58.9% | -17.8 [-29, -7] | .005 | -0.57 |

### Evaluation Bias (Table 4, S02)

| Setting | Condition | F-M | 95% CI | p | d |
|---------|-----------|-----|--------|---|---|
| GPT-5.2 → Gemini 2.0 | Naturalistic | +0.28 | [+0.16, +0.40] | <.001 | +0.82 |
| GPT-5.2 → Gemini 2.0 | Blinded | +0.30 | [+0.17, +0.43] | <.001 | +0.82 |
| Gemini 2.0 → GPT-5.2 | Naturalistic | +0.61 | [+0.48, +0.74] | <.001 | +1.64 |
| Gemini 2.0 → GPT-5.2 | Debiased | +0.03 | [-0.06, +0.13] | .49 | +0.12 |
| Gemini 2.0 → GPT-5.2 | Blinded | +0.44 | [+0.30, +0.59] | <.001 | +1.10 |

## Citation

```bibtex
@article{icke2025gender,
  title={Disentangling Generation and LLM-Judge Effects in Workplace Emails: Gender-Coded Differences Across Models},
  author={Icke, Ilknur},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author Note on AI Assistance

This research was conducted with substantial assistance from Claude Opus 4.5 (Anthropic). The AI assistant contributed to study design, wrote Python code for data collection and analysis, performed statistical computations, and assisted in drafting and revising the manuscript. The human author conceived the research question, supervised all stages, and independently verified all reported statistics against the raw data.

Claude was not included among the models studied. All findings were verified programmatically and the raw data is provided for independent verification.
