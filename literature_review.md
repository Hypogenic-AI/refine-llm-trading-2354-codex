# Literature Review: Refining LLM Trading Toward Long-Horizon Decisions

## Review Scope

### Research Question
Do LLM trading agents perform better when optimized for longer-horizon, position-aware decisions rather than short-term day-to-day profit?

### Inclusion Criteria
- LLM-agent-based trading or financial decision systems
- Benchmarks with sequential decision making (not only static QA)
- Works reporting portfolio/risk metrics (return, Sharpe/Sortino, drawdown)
- Practical code availability or reproducible setup

### Exclusion Criteria
- Pure sentiment classification without trading evaluation
- Static financial QA with no sequential decision loop
- Non-financial agent work without market environment testing

### Time Frame
- Primary focus: 2023-2026

### Sources
- Paper-finder skill output (`paper_search_results/*.jsonl`)
- arXiv PDFs downloaded to `papers/`
- GitHub baseline repositories (`code/`)

## Search Log

| Date | Query | Source | Results | Notes |
|------|-------|--------|---------|------|
| 2026-02-22 | `LLM agents long-term trading decisions finance` | paper-finder | 71 | 7 high-relevance papers (`relevance >= 2`) |
| 2026-02-22 | title-matched arXiv retrieval | arXiv API | 10 downloaded | Added key baseline papers (FinMem/TradingAgents/FinAgent) |

## Screening Results

- Title/abstract screened: 71
- Full papers downloaded: 10
- Deep-read with PDF chunker: 4
  - `StockBench`
  - `FinPos`
  - `Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?` (FINSABER)
  - `Look-Ahead-Bench`

## Key Papers

### 1) StockBench (2025)
- **Authors**: Yanxu Chen et al.
- **Source**: arXiv 2510.02209
- **Key Contribution**: Contamination-aware benchmark for realistic multi-month LLM trading.
- **Methodology**: Daily sequential agent decisions using price/fundamental/news signals.
- **Datasets Used**: Dow Jones stocks with point-in-time style market inputs.
- **Baselines**: Buy-and-hold and multiple frontier LLMs.
- **Metrics**: Cumulative return, maximum drawdown, Sortino ratio, volatility.
- **Key Result**: Most LLM agents fail to consistently beat buy-and-hold; some models improve risk-adjusted behavior.
- **Code**: https://github.com/ChenYXxxx/stockbench
- **Relevance**: Strong evidence that long-horizon evaluation is necessary and harder than short-run tests suggest.

### 2) FinPos (2025)
- **Authors**: Bijia Liu, Ronghao Dang
- **Source**: arXiv 2510.27251
- **Key Contribution**: Position-aware task formulation for continuous portfolio states.
- **Methodology**: Dual-stage decision (direction + risk/size), multi-timescale reward reflection.
- **Datasets Used**: Multi-stock US equities with Yahoo Finance style inputs and macro/news context.
- **Baselines**: Rule-based (MACD/RSI), RL (A2C/DQN/PPO), LLM agents (FinMem/FinCon/TradingAgents/FinAgent).
- **Metrics**: CR, Sharpe, MDD, Calmar.
- **Key Result**: Position-aware architecture improves stability and risk-adjusted outcomes vs one-step agents.
- **Code**: not clearly released in paper text.
- **Relevance**: Directly supports the hypothesis that longer-term, position continuity helps LLM trading systems.

### 3) Can LLM-based Financial Investing Strategies Outperform the Market in Long Run? (FINSABER, 2025)
- **Authors**: Weixian Waylon Li et al.
- **Source**: arXiv 2505.07078
- **Key Contribution**: Long-horizon robustness study over broader universe and longer windows.
- **Methodology**: Two-decade-style backtesting framework with regime analysis and broad symbol coverage.
- **Datasets Used**: US equities universe (S&P 500-centric setup), external financial/news data integrations.
- **Baselines**: Buy-and-hold, factor/rule strategies, RL, and LLM-agent baselines (FinMem/FinAgent/TradingAgents).
- **Metrics**: Sharpe, Sortino, alpha, beta, CR, MDD, volatility.
- **Key Result**: Apparent gains from narrow windows deteriorate under broad long-run testing.
- **Code**: https://github.com/waylonli/FINSABER
- **Relevance**: Core support for the proposed research direction and robust experimental design requirements.

### 4) Look-Ahead-Bench (2026)
- **Author**: Mostapha Benhenda
- **Source**: arXiv 2601.13770
- **Key Contribution**: Standardized benchmark for measuring temporal leakage/look-ahead bias.
- **Methodology**: Practical workflow evaluation with decay analysis across market regimes.
- **Datasets Used**: S&P 500-like setup with point-in-time constraints.
- **Baselines**: Market-cap weighted, momentum, buy-and-hold style quantitative baselines + standard vs PiT LLMs.
- **Metrics**: Alpha decay and regime robustness.
- **Key Result**: Conventional LLMs exhibit stronger look-ahead bias; PiT models are more stable.
- **Code**: https://github.com/benstaf/lookaheadbench
- **Relevance**: Essential for preventing false conclusions in short-term backtests.

### 5) FinMem (2023)
- Layered memory + persona design for LLM trading agents.
- Strong historical baseline for architecture comparisons.

### 6) TradingAgents (2024)
- Open multi-agent framework inspired by trading-firm organization.
- Good engineering baseline for modular long-horizon experiments.

### 7) FinAgent (2024)
- Multimodal (text/chart/numeric) foundation agent with tool augmentation.
- Useful baseline for integrating multiple signal modalities.

### 8) HedgeAgents (2025)
- Hedging-aware multi-agent design for improved robustness during volatile periods.
- Useful for risk-aware portfolio-level comparisons.

### 9) Hierarchical Organization Simulacra (2024)
- Simulates hierarchical investment organizations over long history and many companies.
- Highlights influence of prompt/role design on behavior and profitability.

### 10) Behavioral Consistency Validation (2026)
- Examines whether agent style switching aligns with behavioral finance theory.
- Useful for evaluating realism and stability of long-run agent behavior.

## Common Methodologies

- **Role-based multi-agent decomposition**: TradingAgents, FinPos, HedgeAgents, FinAgent.
- **Memory/reflection loops**: FinMem, FinPos, FinCon-style systems.
- **Point-in-time or contamination-aware evaluation**: StockBench, Look-Ahead-Bench, FINSABER.
- **Hybrid signal fusion**: price + technical indicators + macro/news/fundamentals.

## Standard Baselines

- **Passive**: Buy-and-hold (must include).
- **Rule-based**: MACD, RSI, SMA/WMA crossover, momentum.
- **RL baselines**: A2C/DQN/PPO and FinRL pipelines.
- **LLM-agent baselines**: FinMem, TradingAgents, FinAgent/FinCon-style variants.

## Evaluation Metrics

- **Return metrics**: Cumulative return, annualized return.
- **Risk-adjusted metrics**: Sharpe, Sortino, Calmar.
- **Risk metrics**: Maximum drawdown, volatility.
- **Bias/validity metrics**: Alpha decay/look-ahead robustness by regime.

## Datasets in the Literature

- **US equities (Dow/S&P universes)**: Common in StockBench, FinPos, FINSABER, Look-Ahead-Bench.
- **Price + fundamental + news combinations**: Typical for LLM-agent systems.
- **Sentiment corpora (e.g., finance tweets/news)**: Used for auxiliary signals and LLM adaptation.

## Gaps and Opportunities

1. Many claimed gains are sensitive to short windows, limited symbol sets, or leakage.
2. Position continuity and risk-aware sizing are under-studied relative to directional prediction.
3. Behavioral realism and regime adaptation are not yet standard evaluation requirements.
4. Few studies provide robust out-of-period, multi-regime, and multi-universe comparisons simultaneously.

## Recommendations for Our Experiment

- **Recommended datasets**:
  - `datasets/market_daily_ohlcv_2010_2026/` for long-run multi-asset backtesting.
  - `datasets/twitter_financial_news_sentiment/` for auxiliary sentiment context.
- **Recommended baselines**:
  - Buy-and-hold, MACD/RSI/SMA crossover, FinRL PPO/A2C.
  - LLM-agent baselines adapted from TradingAgents and/or FinMem-style memory policies.
- **Recommended metrics**:
  - CR, Sharpe, Sortino, MDD, Calmar, volatility.
  - Regime-sliced performance and alpha-decay style leakage diagnostics.
- **Methodological considerations**:
  - Enforce point-in-time data boundaries.
  - Use time-based splits and rolling/expanding-window evaluation.
  - Compare daily vs weekly/monthly decision cadence to test the hypothesis directly.
