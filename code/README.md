# Cloned Repositories

## Repo 1: TradingAgents
- URL: https://github.com/TauricResearch/TradingAgents
- Purpose: Multi-agent LLM trading framework with analyst/researcher/risk/trader teams
- Location: `code/tradingagents/`
- Key files:
  - `code/tradingagents/main.py`
  - `code/tradingagents/tradingagents/`
  - `code/tradingagents/cli/main.py`
- Notes:
  - Requires LLM API keys and market data keys.
  - Strong baseline for role-based decision pipelines.

## Repo 2: FinGPT
- URL: https://github.com/AI4Finance-Foundation/FinGPT
- Purpose: Financial LLM ecosystem (sentiment, forecasting, RAG, benchmark)
- Location: `code/fingpt/`
- Key files:
  - `code/fingpt/fingpt/FinGPT_Forecaster/`
  - `code/fingpt/fingpt/FinGPT_Benchmark/`
  - `code/fingpt/fingpt/FinGPT_Sentiment_Analysis_v3/`
- Notes:
  - Useful for sentiment + forecasting modules and benchmark datasets.
  - Includes references to Dow30 forecasting datasets.

## Repo 3: FinRL
- URL: https://github.com/AI4Finance-Foundation/FinRL
- Purpose: Financial RL baseline library with train/test/trade pipeline
- Location: `code/finrl/`
- Key files:
  - `code/finrl/finrl/train.py`
  - `code/finrl/finrl/test.py`
  - `code/finrl/finrl/trade.py`
  - `code/finrl/finrl/meta/env_stock_trading/`
- Notes:
  - Critical non-LLM baseline (PPO/A2C/DDPG etc.) for comparisons.
  - Includes reusable market environments and preprocessing modules.

## Potential Application to This Hypothesis
- Long-horizon LLM strategies can be compared against FinRL baselines on identical data splits.
- TradingAgents/FinGPT can be adapted to weekly or monthly decision cadence.
- FinRL can provide risk-aware benchmark policies and execution environments.
