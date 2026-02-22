# Code Walkthrough

## Code Structure Overview
- `src/run_research.py`
  - Configuration and environment logging
  - Data loading + feature engineering
  - Strategy runners (baselines + LLM)
  - Statistical testing and plotting
  - Output serialization

## Environment Setup
```python
# key libraries
import pandas as pd
import numpy as np
from openai import OpenAI
```

## Key Functions
### `feature_engineering(df)`
Purpose: Build lagged technical features (`ret_1d`, `ret_5d`, `ret_21d`, `vol_21d`, `mom_63d`, `dd_63d`) with anti-leakage shift.

### `llm_allocate(...)`
Purpose: Call real LLM API (`gpt-4.1`) for constrained JSON portfolio weights, with retry and cache.

### `run_rebalance_strategy(...)`
Purpose: Execute rebalance-hold loop for each strategy and compute daily returns with transaction costs.

### `analyze(...)`
Purpose: Run Wilcoxon tests, Shapiro checks, block bootstrap CIs, and effect sizes.

## Data Pipeline
Raw prices -> Lagged features -> Rebalance decisions -> Daily portfolio returns -> Metrics/tests/plots.

## How to Run
```bash
source .venv/bin/activate
python src/run_research.py --model gpt-4.1 --test-start 2025-01-02 --test-end 2026-01-30 --tc-bps 5
```

## Expected Runtime
- First run with API calls: depends on rate limits and network.
- Re-runs: fast due cache (`results/llm_cache.json`).

## Troubleshooting
- Missing API key: set `OPENAI_API_KEY` or `OPENROUTER_API_KEY`.
- If `uv add` fails for this workspace: use `uv pip install` and update `requirements.txt`.
