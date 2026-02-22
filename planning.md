# Planning: Refining LLM Trading Toward Long-Horizon Decisions

## Motivation & Novelty Assessment

### Why This Research Matters
Most LLM trading agents are evaluated on dense daily actions, which can overfit noise and produce high turnover with weak risk-adjusted performance. If longer-horizon decisions are superior, practitioners can reduce trading frequency, transaction costs, and behavioral instability while improving real deployability. This directly affects retail quant tooling, institutional prototyping, and benchmark design for financial LLM agents.

### Gap in Existing Work
The literature shows that many short-window gains degrade in realistic long-run tests (StockBench, FINSABER), and position-aware formulations help stability (FinPos). However, a controlled experiment that isolates **decision cadence** (daily vs weekly vs monthly) under the same LLM model, same universe, and same prompt family is still limited.

### Our Novel Contribution
We test a single real LLM policy under matched conditions while varying only rebalance horizon and position-awareness. This provides a clean estimate of whether long-horizon decision-making improves risk-adjusted outcomes and trading stability relative to short-term optimization.

### Experiment Justification
- Experiment 1: Daily vs weekly vs monthly LLM rebalancing.
  Why needed: Directly tests the core hypothesis by isolating decision horizon.
- Experiment 2: Position-aware vs memoryless LLM policy at weekly cadence.
  Why needed: Tests whether explicit continuity is a mechanism behind long-horizon gains.
- Experiment 3: Baseline comparison (equal-weight buy-and-hold, momentum, volatility-parity).
  Why needed: Ensures LLM results are interpreted against practical non-LLM alternatives.

## Research Question
Do LLM trading agents perform better, on risk-adjusted and drawdown-aware metrics, when making longer-horizon portfolio decisions (weekly/monthly) than short-term daily decisions?

## Background and Motivation
Recent papers in this workspace (StockBench 2025, FinPos 2025, FINSABER 2025, Look-Ahead-Bench 2026) indicate that naive daily LLM trading often fails to beat passive baselines and may exploit leakage or short-window noise. Long-horizon and position-aware methods appear more robust, but controlled cadence studies are sparse. This project targets that methodological gap with a reproducible, point-in-time simulation using local market data from 2010-2026.

## Hypothesis Decomposition
- H1 (comparative): Weekly and monthly LLM strategies have higher Sortino ratio than daily LLM strategy on 2024-2026 test data.
- H2 (risk): Weekly/monthly LLM strategies have lower maximum drawdown and turnover than daily LLM.
- H3 (mechanistic): Position-aware prompting improves weekly strategy stability (lower turnover, lower drawdown) versus memoryless prompting.
- H0 (null): No statistically significant difference in risk-adjusted performance across decision cadences.

Independent variables:
- Rebalance cadence: 1D, 5D, 21D.
- Prompt style: position-aware vs memoryless.

Dependent variables:
- Return metrics: cumulative return, annualized return.
- Risk-adjusted metrics: Sharpe, Sortino, Calmar.
- Risk/stability metrics: max drawdown, annualized volatility, turnover.

Success criteria:
- Support for hypothesis if at least one long-horizon cadence (5D or 21D) beats daily by >=0.20 Sortino and shows lower turnover, with consistent directional advantage in bootstrap CI.

## Proposed Methodology

### Approach
Use a single LLM (OpenAI GPT-4.1) as a portfolio allocator over a fixed US equity universe (15 tickers). At each rebalance date, provide only lagged technical features and current positions (for position-aware runs), require strict JSON allocations, and execute next-period returns with transaction-cost penalty.

### Experimental Steps
1. Validate and preprocess local OHLCV data; compute lagged features per ticker.
   Rationale: ensure no look-ahead leakage and stable feature inputs.
2. Implement non-LLM baselines (equal-weight buy-and-hold, momentum top-k, inverse-volatility).
   Rationale: contextualize LLM outcomes against realistic standards.
3. Implement LLM allocation engine with response caching.
   Rationale: reproducibility, cost control, robust retry handling.
4. Run Experiment 1 (cadence comparison) over identical test period.
   Rationale: direct test of hypothesis.
5. Run Experiment 2 (position-aware ablation) at weekly cadence.
   Rationale: evaluate mechanism for long-horizon improvement.
6. Compute metrics, statistical tests, and visualizations.
   Rationale: quantify significance and practical effect.

### Baselines
- Equal-weight buy-and-hold (monthly rebalance only for cash normalization).
- Momentum top-5 (63-day momentum, same cadence grid).
- Inverse-volatility parity (21-day volatility, same cadence grid).

### Evaluation Metrics
- Cumulative return (CR), annualized return (AnnRet).
- Sharpe, Sortino, Calmar.
- Maximum drawdown (MDD), annualized volatility.
- Turnover and number of trades.

Why: These match literature standards and directly measure risk-adjusted long-horizon quality.

### Statistical Analysis Plan
- Primary comparison: paired daily return differences between strategies on common dates.
- Tests:
  - Wilcoxon signed-rank (non-normal robustness).
  - Circular block bootstrap (95% CI) for Sortino difference.
- Significance threshold: alpha = 0.05.
- Effect size: Cliff's delta on daily return differences.

## Expected Outcomes
- If hypothesis holds: weekly/monthly LLM policies outperform daily LLM on Sortino and MDD, with lower turnover.
- If partially supported: returns may be similar, but long-horizon strategies still win on stability/risk metrics.
- If refuted: daily LLM remains best risk-adjusted policy, suggesting signal half-life is very short.

## Timeline and Milestones
- Phase 0-1 planning: completed in this session.
- Phase 2 setup + data validation: ~20 min.
- Phase 3 implementation: ~60 min.
- Phase 4 experiment runs with API calls: ~60-90 min (with caching).
- Phase 5 analysis + plots: ~30 min.
- Phase 6 reporting + validation: ~30 min.

## Potential Challenges
- API rate limits or malformed JSON outputs.
  Mitigation: retry/backoff, schema checks, cache + fallback equal-weight output.
- Regime dependency due to limited test years.
  Mitigation: sub-period analysis and bootstrap CIs.
- Transaction cost assumptions influence absolute performance.
  Mitigation: fixed cost sensitivity checks.

## Success Criteria
- End-to-end reproducible pipeline runs from raw data to report.
- Results include real LLM API outputs, not simulated decisions.
- REPORT.md provides full metrics table, stats tests, plots, and limitations.
