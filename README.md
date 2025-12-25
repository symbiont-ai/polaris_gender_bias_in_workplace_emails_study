# Disentangling Generation and LLM-Judge Effects in Workplace Emails: Gender-Coded Differences Across Models

This repository contains the code, data, and paper for our study on gender bias in LLM-generated workplace emails and their evaluation by other LLMs.

## Key Findings

1. **Context-dependent effects**: Strong effects emerged in credit attribution (S02); in salary negotiation (S01) we observe no evaluation bias and only a limited generation-style difference (Gemini: “I believe”)
2. **Different models encode different stereotypes**: GPT-5.2 generates female emails with softer framing; Gemini 2.0 generates female emails with collaborative framing
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
│   └── analyze_results.py       # Statistical analysis
├── data/
│   ├── raw/                     # Original JSON files
│   │   ├── emails_gpt52.json    # 360 GPT-5.2 generated emails
│   │   ├── emails_gemini.json   # 360 Gemini 2.0 generated emails
│   │   └── ratings_*.json       # All ratings (6 files)
│   └── processed/               # Analysis outputs (CSVs)
├── notebooks/
│   └── reproduce_tables.ipynb   # Jupyter notebook reproducing all tables
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick Start

### Reproduce the Analysis

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) if regenerating emails/ratings, set API keys
cp .env.example .env
# then fill in OPENAI_API_KEY and GEMINI_API_KEY

# Run analysis script (persona-level, as in paper)
cd src
python analyze_results.py --data-dir ../data/raw --output-dir ../data/processed

# Compare persona-level vs email-level analysis
python analyze_results.py --compare

# Run email-level analysis (NOT recommended, shown for transparency)
python analyze_results.py --email-level
```

Or use the Jupyter notebook:
```bash
cd notebooks
jupyter notebook reproduce_tables.ipynb
```

### Why Persona-Level Analysis?

Each persona generates 3 emails. The naive approach treats these as 90 independent observations per gender, but this **inflates sample size** and **underestimates standard errors**.

The correct approach aggregates to persona-level means first (n=30 per gender), respecting the clustering structure. Use `--compare` to see the difference:

```
Pattern              Level           n      p-value
GPT-5.2 full_name_sig Persona        30     0.002197  ← CORRECT
                      Email          90     0.000210  (inflated n)
```

### Reproduce Data Collection (requires API keys)

```bash
# Set API keys
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"

# Generate emails
cd src
python generate_emails.py --model gpt-5.2 --output ../data/raw/emails_gpt52.json
python generate_emails.py --model gemini --output ../data/raw/emails_gemini.json

# Rate emails
python rate_emails.py --emails ../data/raw/emails_gpt52.json --evaluator gemini --output ../data/raw/ratings_gemini_naturalistic.json
# ... (repeat for other conditions)
```

## Methodology

### Design
- 30 matched persona pairs (60 total) differing only in gendered names
- 2 scenarios: salary negotiation (S01), credit attribution (S02)
- 3 emails per persona-scenario combination
- Cross-model evaluation: GPT-5.2 ↔ Gemini 2.0

### Statistical Approach
- **Unit of analysis**: Persona (n=30 per gender), not individual emails
- **Test**: Mann-Whitney U (non-parametric)
- **Effect size**: Cohen's d
- **Multiple comparisons**: Benjamini-Hochberg FDR correction (α=0.05)

### Conditions
1. **Naturalistic**: Standard evaluation prompt
2. **Debiased**: Added "Be objective and consistent. Focus only on the content of the email, not on any assumptions about the sender."
3. **Blinded**: Names replaced with [SENDER]

## Data Format

### Emails (JSON)
```json
{
  "persona_id": "pair_01_F",
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
  "persona_id": "pair_01_F",
  "gender": "F",
  "scenario_id": "S02",
  "likelihood_to_send_correction": 4,
  "professionalism": 5,
  "perceived_reasonableness": 4
}
```

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
