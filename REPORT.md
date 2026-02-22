# REPORT: Refining LLM Trading

## 1. Executive Summary
This study tests whether LLM trading agents perform better when making longer-horizon decisions instead of day-to-day reallocation. Using real GPT-4.1 API calls on US equity daily data (15 tickers, 2010-2026), we compared daily, weekly, and monthly LLM portfolio policies under identical features and constraints.

Key finding: longer-horizon LLM policies outperformed daily LLM on risk-adjusted metrics in this test period (2025-01-02 to 2026-01-30), with weekly cadence showing the strongest LLM result (Sortino 2.84 vs 2.17 daily, lower drawdown, much lower turnover).

Practical implication: for LLM-based portfolio control, reducing action frequency appears to improve stability and net performance under transaction costs, though in this setup a non-LLM momentum baseline still ranked best overall.

## 2. Goal
### Hypothesis
LLM agents may perform better in finance by making longer-term trading decisions rather than optimizing short-term day-to-day profit.

### Why Important
Most LLM trading papers report short-horizon decisions. Prior work in this workspace (StockBench, FinPos, FINSABER, Look-Ahead-Bench) indicates short-window gains may not survive realistic evaluation. A direct cadence-controlled experiment is needed.

### Problem Solved
This work isolates decision cadence while holding model, dataset, and prompt family constant, producing an apples-to-apples estimate of short vs long horizon behavior.

### Expected Impact
Guides benchmark design and practical deployment: lower-frequency LLM control may reduce churn and improve deployability.

## 3. Data Construction
### Dataset Description
Primary market dataset:
- Source: Yahoo Finance via pre-gathered dataset `datasets/market_daily_ohlcv_2010_2026/prices.parquet`
- Size: 60,660 rows, 15 tickers, date range 2010-01-04 to 2026-01-30
- Fields: `Date, Open, High, Low, Close, Adj Close, Volume, ticker`

Auxiliary sentiment dataset (not used directly in allocation features, retained for future extension):
- Source: HuggingFace `zeroshot/twitter-financial-news-sentiment`
- Size: train 9,543 / validation 2,388
- Fields: `text, label`

### Example Samples
Market sample rows:

| Date | Open | High | Low | Close | Adj Close | Volume | ticker |
|---|---:|---:|---:|---:|---:|---:|---|
| 2018-12-10 | 178.60 | 179.84 | 174.68 | 178.83 | 151.93 | 4,528,300 | GS |
| 2014-09-26 | 99.44 | 100.64 | 99.24 | 100.38 | 74.99 | 2,935,900 | CAT |
| 2022-08-19 | 351.00 | 351.86 | 347.50 | 349.27 | 319.40 | 1,812,300 | GS |

Sentiment sample rows:

| split | text (truncated) | label |
|---|---|---:|
| train | `$BYND - JPMorgan reels in expectations on Beyond Meat ...` | 0 |
| train | `$CCL $RCL - Nomura points to bookings weakness ...` | 0 |
| validation | `$ALLY - Ally Financial pulls outlook ...` | 0 |

### Data Quality
From `results/data_quality.json`:
- Missing values (raw OHLCV): 0%
- Duplicates: 0
- Feature missingness only from rolling-window warmup:
  - `ret_1d`: 0.049%
  - `ret_5d`: 0.148%
  - `ret_21d`: 0.544%
  - `vol_21d`: 0.544%
  - `mom_63d`: 1.583%
  - `dd_63d`: 1.558%

### Preprocessing Steps
1. Renamed `Date -> date`, `Adj Close -> adj_close`; sorted by (`ticker`, `date`).
2. Computed lagged technical features per ticker:
   - `ret_1d`, `ret_5d`, `ret_21d`, `vol_21d`, `mom_63d`, `dd_63d`.
3. Shifted all features by 1 day to prevent look-ahead.
4. Built return panel from `adj_close.pct_change()` for backtesting.

### Train/Val/Test Splits
- Historical context: full 2010-2026 available.
- Main evaluation window (fixed test): 2025-01-02 to 2026-01-30 (270 trading days).
- Rationale: keep enough daily decisions for cadence comparison while staying in recent out-of-sample period.

## 4. Experiment Description
### Methodology
#### High-Level Approach
At each rebalance date, the strategy receives lagged per-ticker features and outputs long-only weights. We compare:
- LLM daily rebalance (short horizon)
- LLM weekly rebalance
- LLM monthly rebalance
- Weekly position-aware vs weekly memoryless LLM ablation
- Non-LLM baselines

#### Why This Method?
It isolates cadence as the independent variable while controlling model and input schema. This directly targets the user’s question.

Alternatives considered:
- Full RL retraining per horizon: rejected for this session due longer training cycle and confounded architecture changes.
- Multi-model comparison first: deferred; cadence isolation prioritized.

### Implementation Details
#### Tools and Libraries
- Python 3.12.2
- pandas 2.3.1
- numpy 2.3.5
- scipy 1.17.0
- matplotlib 3.10.8
- seaborn 0.13.2
- openai 2.21.0

#### Algorithms/Models
- LLM: `gpt-4.1` via OpenAI API (real model calls, cached)
- Baselines:
  - Equal-weight
  - Momentum top-k (63-day momentum, weekly)
  - Inverse-volatility parity (21-day vol, weekly)

#### Hyperparameters
| Parameter | Value | Selection Method |
|---|---:|---|
| model | gpt-4.1 | fixed SOTA API model |
| temperature | 0.0 | deterministic reproducibility |
| max_weight_per_asset | 0.35 | risk cap |
| top_k_hint | 5 | portfolio concentration prior |
| transaction_cost | 5 bps | realistic friction baseline |
| bootstrap_samples | 1000 | standard CI stability |
| bootstrap_block | 5 days | preserves short autocorrelation |

#### Training / Analysis Pipeline
1. Compute lagged features.
2. At rebalance date, request JSON weights from LLM.
3. Apply transaction cost on rebalance day using turnover.
4. Hold weights until next rebalance.
5. Aggregate daily strategy returns and compute metrics/statistics.

### Experimental Protocol
#### Reproducibility Information
- Number of runs: 1 primary + repeated reruns for reproducibility and cost sensitivity
- Seeds: 42 (primary), deterministic temperature 0
- Hardware: 2x NVIDIA RTX 3090 (24GB each), API-based workload (GPU unused for model inference)
- Execution profile: ~391 LLM rebalances in main run
- LLM token usage (main run): 752,721 total tokens (`results/usage_summary.json`)

#### Evaluation Metrics
- `cum_return`: total portfolio growth over test.
- `ann_return`: geometric annualized return.
- `ann_vol`: annualized daily-return volatility.
- `sharpe`: mean/std annualized.
- `sortino`: mean/downside-std annualized.
- `max_drawdown`: worst peak-to-trough drawdown.
- `calmar`: annualized return / |max drawdown|.
- `turnover`: sum absolute weight changes across rebalances.

### Raw Results
#### Main table (5 bps transaction costs)
| Method | CumReturn | AnnRet | Sharpe | Sortino | MaxDD | Turnover |
|---|---:|---:|---:|---:|---:|---:|
| momentum_weekly | 0.4527 | 0.4169 | 2.2372 | 2.9801 | -0.1613 | 16.9333 |
| llm_weekly_posaware | 0.4261 | 0.3928 | 2.0966 | 2.8352 | -0.1518 | 24.2000 |
| llm_weekly_memoryless | 0.4274 | 0.3940 | 2.0820 | 2.7305 | -0.1660 | 37.2333 |
| llm_monthly_posaware | 0.3804 | 0.3511 | 1.8274 | 2.4603 | -0.1347 | 12.0333 |
| llm_daily_posaware | 0.3091 | 0.2858 | 1.6095 | 2.1650 | -0.1432 | 59.4933 |
| equal_weight | 0.2841 | 0.2629 | 1.4577 | 1.8574 | -0.1897 | 0.0000 |
| invvol_weekly | 0.2260 | 0.2094 | 1.3125 | 1.7189 | -0.1752 | 6.3687 |

#### Robustness: transaction cost sensitivity
Weekly/monthly LLM stayed above daily LLM at 0, 5, and 10 bps (`results/metrics_tc0.csv`, `results/metrics_tc5.csv`, `results/metrics_tc10.csv`).

#### Output Locations
- Results JSON: `results/metrics.json`
- Metrics CSV: `results/metrics.csv`
- Daily returns: `results/daily_returns.csv`
- Statistical tests: `results/stat_tests.json`
- Usage summary: `results/usage_summary.json`
- Plots: `figures/equity_curves.png`, `figures/drawdown_curves.png`, `figures/return_boxplot.png`

## 5. Result Analysis
### Key Findings
1. **Longer-horizon LLM beats daily LLM on risk-adjusted performance**.
   - Weekly LLM Sortino 2.835 vs Daily LLM 2.165.
   - Monthly LLM Sortino 2.460 vs Daily LLM 2.165.
2. **Daily LLM is much higher turnover**.
   - Daily turnover 59.49 vs Weekly 24.20 vs Monthly 12.03.
3. **Position-awareness mainly helps stability, not raw return**.
   - Weekly pos-aware MDD -0.1518 vs memoryless -0.1660.
   - Weekly pos-aware turnover 24.20 vs memoryless 37.23.
4. **A tuned non-LLM baseline (momentum weekly) still ranked highest** in this window.

### Hypothesis Testing Results
Null hypothesis H0: no difference between long-horizon and daily LLM outcomes.

From `results/stat_tests.json`:
- Weekly pos-aware vs Daily pos-aware:
  - Wilcoxon p = 0.0272 (significant at 0.05)
  - Mean daily diff = +0.000318
  - Sortino diff bootstrap CI95 = [-0.355, 2.073] (wide, includes 0)
  - Cliff’s delta = 0.0123 (small effect)
- Monthly pos-aware vs Daily pos-aware:
  - Wilcoxon p = 0.4382 (not significant)
- Weekly pos-aware vs Weekly memoryless:
  - Wilcoxon p = 0.5787 (not significant on daily return level)

Interpretation:
- Evidence supports weekly over daily on rank-based daily return test and clear practical metrics (Sortino/turnover).
- Monthly improvement is practical but not statistically significant in this window.

### Comparison to Baselines
- Weekly LLM outperformed equal-weight and inverse-volatility baselines on all key risk-adjusted metrics.
- Weekly momentum baseline still outperformed LLM (Sortino 2.98 vs 2.84).

### Visualizations
- `figures/equity_curves.png`: cumulative wealth trajectories.
- `figures/drawdown_curves.png`: drawdown severity over time.
- `figures/return_boxplot.png`: distributional comparison of daily returns.

### Surprises and Insights
- Memoryless weekly LLM matched pos-aware weekly cumulative return closely, but with much higher turnover.
- Daily LLM underperformed weekly/monthly despite more frequent opportunities, consistent with noise-chasing concerns.

### Error Analysis
Failure modes observed in trade logs:
- Occasional weight concentration persistence during trend regimes.
- Weekly decisions often repeated unchanged allocations for multiple rebalance points.
  - Example: 2025-01-02 and 2025-01-10 had identical top-5 allocations for weekly pos-aware run.
- This behavior reduced churn but may miss abrupt reversals.

### Limitations
- Single model family (`gpt-4.1`) and single equity universe (15 tickers).
- Feature set is technical-only; no company-specific news stream fused into daily prompts.
- Test horizon is recent and limited (2025-2026 Jan), so regime diversity is moderate.
- No explicit market impact/slippage model beyond proportional transaction costs.
- Bootstrap CI for Sortino differences is wide; more years/markets needed for stronger inference.

## 6. Conclusions
### Summary
In this controlled experiment, LLM trading performance improved when decisions were made less frequently (weekly/monthly) versus daily reallocation. The strongest LLM variant was weekly position-aware, which delivered better risk-adjusted performance and lower turnover than daily LLM.

### Implications
- Practical: LLM portfolio agents may be better used as medium-horizon allocators rather than day-trading controllers.
- Methodological: cadence should be a standard ablation axis in LLM trading papers.

### Confidence in Findings
Moderate confidence. Results are reproducible in this workspace and robust across transaction-cost sensitivity, but broader validation across assets/periods/models is still needed.

## 7. Next Steps
### Immediate Follow-ups
1. Add a second model (`gpt-5` or Claude Sonnet 4.5) for cross-model cadence generalization.
2. Add explicit news/fundamental context with strict point-in-time joins.
3. Expand test to rolling windows across 2016-2026 for regime-stratified inference.

### Alternative Approaches
- Compare LLM cadence policies to RL policies trained per cadence in same environment.
- Use hierarchical agent setup (analyst + risk manager + executor) from TradingAgents architecture.

### Broader Extensions
- Apply cadence study to crypto and futures portfolios.
- Test policy transfer from US equities to international equity universes.

### Open Questions
- Is the weekly advantage persistent under macro shock periods (e.g., 2020-like volatility)?
- Does position-awareness matter more when transaction costs increase further?

## 8. Validation Checklist
### Code Validation
- [x] Full script runs without errors.
- [x] Re-ran experiment multiple times (identical `tc=5` outputs).
- [x] Random seed and deterministic sampling set.
- [x] No hardcoded absolute paths.
- [x] No look-ahead in feature construction (all shifted by 1 day).

### Scientific Validation
- [x] Non-parametric tests used due non-normal return differences.
- [x] Assumption checks included (Shapiro p-values reported).
- [x] Alternative explanations discussed.
- [x] Limitations documented.

### Documentation Validation
- [x] All required sections included.
- [x] Plot paths and output files documented.
- [x] Reproducibility instructions provided in README.

## 9. References
- Workspace literature review: `literature_review.md`
- Resources catalog: `resources.md`
- StockBench (2025), FinPos (2025), FINSABER (2025), Look-Ahead-Bench (2026), FinMem (2023), TradingAgents (2024), FinAgent (2024)
