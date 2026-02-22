# Resources Catalog

## Summary
This document catalogs all gathered resources for the project **Refining LLM Trading**, including papers, datasets, and code repositories.

## Papers
Total papers downloaded: 10

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| StockBench | Chen et al. | 2025 | `papers/2510.02209v1_StockBench_Can_LLM_Agents_Trade_Stocks_Profitably_In_Real_world_Market.pdf` | Multi-month benchmark; buy-and-hold remains hard to beat |
| FinPos | Liu, Dang | 2025 | `papers/2510.27251v2_FinPos_A_Position_Aware_Trading_Agent_System_for_Real_Financial_Market.pdf` | Position-aware long-horizon LLM trading |
| Hierarchical Organization Simulacra | Chen et al. | 2024 | `papers/2410.00354v1_Hierarchical_Organization_Simulacra_in_the_Investment_Sector.pdf` | Multi-agent hierarchy over long history |
| Can LLM-based ... Outperform ... Long Run? | Li et al. | 2025 | `papers/2505.07078v5_Can_LLM_based_Financial_Investing_Strategies_Outperform_the_Market_in_.pdf` | Long-horizon robustness and survivorship-bias critique |
| HedgeAgents | Li et al. | 2025 | `papers/2502.13165v1_HedgeAgents_A_Balanced_aware_Multi_agent_Financial_Trading_System.pdf` | Hedging-aware architecture |
| Look-Ahead-Bench | Benhenda | 2026 | `papers/2601.13770v1_Look_Ahead_Bench_a_Standardized_Benchmark_of_Look_ahead_Bias_in_Point_.pdf` | Temporal leakage benchmark |
| Behavioral Consistency Validation | Li et al. | 2026 | `papers/2602.07023v1_Behavioral_Consistency_Validation_for_LLM_Agents_An_Analysis_of_Tradin.pdf` | Behavioral-finance consistency checks |
| FinMem | Yu et al. | 2023 | `papers/2311.13743v2_FinMem_A_Performance_Enhanced_LLM_Trading_Agent_With_Layered_Memory_an.pdf` | Layered memory baseline |
| TradingAgents | Xiao et al. | 2024 | `papers/2412.20138v7_TradingAgents_Multi_Agents_LLM_Financial_Trading_Framework.pdf` | Open-source multi-agent baseline |
| FinAgent | Zhang et al. | 2024 | `papers/2402.18485v3_A_Multimodal_Foundation_Agent_for_Financial_Trading_Tool_Augmented_Div.pdf` | Multimodal foundation trading agent |

See `papers/README.md` and `papers/papers_metadata.json` for details.

## Datasets
Total datasets downloaded: 2

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| market_daily_ohlcv_2010_2026 | Yahoo Finance (`yfinance`) | 60,660 rows, 15 tickers | Long-horizon trading/backtest | `datasets/market_daily_ohlcv_2010_2026/` | Includes `prices.parquet` + sample CSV |
| twitter_financial_news_sentiment | HuggingFace `zeroshot/twitter-financial-news-sentiment` | Train 9,543 / Val 2,388 | Sentiment feature extraction | `datasets/twitter_financial_news_sentiment/` | Saved to disk + sample JSON |

See `datasets/README.md` and `datasets/dataset_stats.json`.

## Code Repositories
Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| TradingAgents | https://github.com/TauricResearch/TradingAgents | LLM multi-agent trading framework | `code/tradingagents/` | Practical baseline architecture |
| FinGPT | https://github.com/AI4Finance-Foundation/FinGPT | Financial LLM ecosystem (sentiment, forecasting, benchmark) | `code/fingpt/` | Useful modules + datasets references |
| FinRL | https://github.com/AI4Finance-Foundation/FinRL | RL trading baselines and environments | `code/finrl/` | Strong non-LLM baseline suite |

See `code/README.md`.

## Resource Gathering Notes

### Search Strategy
- Used paper-finder (`find_papers.py`) as primary search method.
- Filtered high-priority papers by relevance score (`>=2`).
- Added high-impact baseline papers required for experiment design.
- Downloaded all selected papers as PDFs into `papers/`.

### Selection Criteria
- Direct relevance to long-horizon vs short-horizon LLM trading.
- Availability of reproducible methodology and benchmarks.
- Presence of robust metrics and risk-aware evaluation.
- Open-source code preferred.

### Challenges Encountered
- `uv add` failed because this repo is not packaged for editable install; used `uv pip install` fallback.
- HF dataset `financial_phrasebank` was script-based and unsupported in current `datasets` package; used script-free alternative (`twitter-financial-news-sentiment`).

### Gaps and Workarounds
- Some newest papers provide limited implementation details; supplemented with open-source framework repos (TradingAgents, FinGPT, FinRL).
- For robust long-horizon experiments, point-in-time and leakage controls must be enforced during experiment-runner phase.

## Recommendations for Experiment Design

1. **Primary dataset(s)**: `market_daily_ohlcv_2010_2026` for sequential long-horizon evaluation; augment with `twitter_financial_news_sentiment` for sentiment signals.
2. **Baseline methods**: Buy-and-hold, MACD/RSI/SMA crossover, FinRL PPO/A2C, TradingAgents default workflow.
3. **Evaluation metrics**: CR, Sharpe, Sortino, MDD, Calmar, volatility, plus regime-sliced performance and leakage checks.
4. **Code to adapt/reuse**: `code/tradingagents/` for LLM-agent orchestration, `code/finrl/` for RL baselines/environments, `code/fingpt/` for finance-specific LLM modules.

## Research Execution Log (2026-02-22)

### What Was Executed
- Implemented full experiment runner: `src/run_research.py`
- Ran real API-based LLM experiments with `gpt-4.1`:
  - `llm_daily_posaware`
  - `llm_weekly_posaware`
  - `llm_monthly_posaware`
  - `llm_weekly_memoryless` (ablation)
- Ran non-LLM baselines:
  - `equal_weight`
  - `momentum_weekly`
  - `invvol_weekly`
- Generated outputs:
  - `results/metrics.csv`, `results/stat_tests.json`, `results/daily_returns.csv`
  - `figures/equity_curves.png`, `figures/drawdown_curves.png`, `figures/return_boxplot.png`

### Reproducibility Notes
- Environment: `.venv` in workspace root.
- Dependency install method: `uv pip install` fallback due workspace build packaging issue.
- Deterministic settings: `seed=42`, `temperature=0.0`.
- LLM response caching: `results/llm_cache.json`.
- Cost/robustness runs completed for transaction costs `0/5/10` bps (`results/metrics_tc*.csv`).
