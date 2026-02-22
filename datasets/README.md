# Downloaded Datasets

This directory contains datasets for experiments on long-horizon LLM trading.
Large data files are intentionally excluded from git by `datasets/.gitignore`.

## Dataset 1: market_daily_ohlcv_2010_2026

### Overview
- Source: Yahoo Finance via `yfinance`
- Size: 60,660 rows across 15 tickers
- Format: Parquet (`prices.parquet`) + CSV sample
- Task: Daily portfolio/trading decision simulation
- Splits: Time-based split recommended (e.g., train 2010-2020, val 2021-2023, test 2024-2026)
- License: Yahoo Finance terms

### Download Instructions

Using Python (recommended):
```python
import pandas as pd
import yfinance as yf

symbols = ["AAPL","MSFT","JPM","GS","V","JNJ","WMT","DIS","CAT","IBM","KO","MCD","NKE","NVDA","SPY"]
frames = []
for s in symbols:
    df = yf.download(s, start="2010-01-01", end="2026-01-31", auto_adjust=False, progress=False)
    df = df.reset_index()
    df["ticker"] = s
    frames.append(df[["Date","Open","High","Low","Close","Adj Close","Volume","ticker"]])
prices = pd.concat(frames, ignore_index=True)
prices.to_parquet("datasets/market_daily_ohlcv_2010_2026/prices.parquet", index=False)
```

### Loading the Dataset
```python
import pandas as pd
prices = pd.read_parquet("datasets/market_daily_ohlcv_2010_2026/prices.parquet")
```

### Sample Data
- `datasets/market_daily_ohlcv_2010_2026/sample_100_rows.csv`

### Notes
- Directly aligned with long-horizon backtesting and portfolio decisions.
- Includes market regimes (post-2010 bull/bear/recovery periods).

## Dataset 2: twitter_financial_news_sentiment

### Overview
- Source: HuggingFace `zeroshot/twitter-financial-news-sentiment`
- Size: train 9,543 / validation 2,388
- Format: HuggingFace dataset saved to disk + JSON sample
- Task: Financial text sentiment signal extraction for trading agent context
- Splits: Provided by source dataset (train/validation)
- License: See dataset card on HuggingFace

### Download Instructions

Using HuggingFace:
```python
from datasets import load_dataset

ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
ds.save_to_disk("datasets/twitter_financial_news_sentiment/hf_dataset")
```

### Loading the Dataset
```python
from datasets import load_from_disk

ds = load_from_disk("datasets/twitter_financial_news_sentiment/hf_dataset")
```

### Sample Data
- `datasets/twitter_financial_news_sentiment/samples/samples.json`

### Notes
- Useful for building sentiment features to combine with price signals.
- Best used as auxiliary signal, not standalone trading objective.

## Validation Summary
- Dataset stats: `datasets/dataset_stats.json`
- Quick checks completed: file existence, row counts, schema, and sample export.
