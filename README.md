# Refining LLM Trading

This project tests whether LLM trading agents perform better with longer-horizon decisions than day-to-day rebalancing. We run real GPT-4.1 API-based portfolio allocation on local US equity data and compare daily, weekly, and monthly cadences with non-LLM baselines.

## Key Findings
- Weekly and monthly LLM strategies outperformed daily LLM on risk-adjusted metrics in the 2025-01-02 to 2026-01-30 test window.
- Weekly LLM (position-aware) achieved Sortino `2.835` vs daily LLM `2.165`, with much lower turnover (`24.20` vs `59.49`).
- A weekly momentum baseline still ranked best overall (Sortino `2.980`).
- Results were stable under transaction cost sensitivity (`0/5/10` bps).

## Reproduce
```bash
# from workspace root
source .venv/bin/activate
python src/run_research.py --model gpt-4.1 --test-start 2025-01-02 --test-end 2026-01-30 --tc-bps 5
```

Optional sensitivity runs:
```bash
python src/run_research.py --model gpt-4.1 --test-start 2025-01-02 --test-end 2026-01-30 --tc-bps 0
cp results/metrics.csv results/metrics_tc0.csv
python src/run_research.py --model gpt-4.1 --test-start 2025-01-02 --test-end 2026-01-30 --tc-bps 10
cp results/metrics.csv results/metrics_tc10.csv
```

## File Structure
- `planning.md`: phase 0/1 motivation, novelty, and experimental plan.
- `src/run_research.py`: full experiment runner (data prep + backtests + stats + plots).
- `REPORT.md`: full scientific report with methods, results, and limitations.
- `results/`: metrics, daily returns, tests, cache, usage summary.
- `figures/`: equity/drawdown/distribution plots.
- `literature_review.md`, `resources.md`: pre-gathered context and resource catalog.
