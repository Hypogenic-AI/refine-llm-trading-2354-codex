import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from openai import OpenAI
from scipy.stats import shapiro, wilcoxon


@dataclass
class Config:
    seed: int = 42
    model: str = "gpt-4.1"
    temperature: float = 0.0
    max_weight: float = 0.35
    top_k_hint: int = 5
    tc_bps: float = 5.0
    test_start: str = "2025-01-02"
    test_end: str = "2026-01-30"
    price_path: str = "datasets/market_daily_ohlcv_2010_2026/prices.parquet"
    output_dir: str = "results"
    figures_dir: str = "figures"
    cache_path: str = "results/llm_cache.json"
    n_bootstrap: int = 1000
    bootstrap_block: int = 5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs(cfg: Config) -> None:
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.figures_dir).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)


def build_client() -> OpenAI:
    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openai_key:
        return OpenAI(api_key=openai_key)
    if openrouter_key:
        return OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
    raise RuntimeError("Neither OPENAI_API_KEY nor OPENROUTER_API_KEY is set.")


def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "ticker" not in df.columns:
        raise ValueError("Expected 'ticker' column in prices dataset.")
    df = df.rename(columns={"Date": "date", "Adj Close": "adj_close"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for ticker, g in df.groupby("ticker", sort=False):
        g = g.copy()
        g["ret_1d"] = g["adj_close"].pct_change()
        g["ret_5d"] = g["adj_close"].pct_change(5)
        g["ret_21d"] = g["adj_close"].pct_change(21)
        g["vol_21d"] = g["ret_1d"].rolling(21).std()
        g["mom_63d"] = g["adj_close"].pct_change(63)
        roll_max = g["adj_close"].rolling(63).max()
        g["dd_63d"] = g["adj_close"] / roll_max - 1.0
        for c in ["ret_1d", "ret_5d", "ret_21d", "vol_21d", "mom_63d", "dd_63d"]:
            g[c] = g[c].shift(1)
        out.append(g)
    feat = pd.concat(out, ignore_index=True)
    return feat


def data_quality_checks(df: pd.DataFrame) -> Dict:
    missing = df.isna().mean().to_dict()
    dups = int(df.duplicated(subset=["ticker", "date"]).sum())
    desc = {
        "rows": int(len(df)),
        "tickers": int(df["ticker"].nunique()),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "missing_rate": {k: float(v) for k, v in missing.items()},
        "duplicate_rows": dups,
    }
    return desc


def prepare_panels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = ["ret_1d", "ret_5d", "ret_21d", "vol_21d", "mom_63d", "dd_63d"]
    feat_panel = df.pivot(index="date", columns="ticker", values=features)
    prices = df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    rets = prices.pct_change().fillna(0.0)
    return feat_panel, prices, rets


def annualized_return(daily_returns: pd.Series) -> float:
    if len(daily_returns) == 0:
        return 0.0
    total = (1 + daily_returns).prod()
    years = len(daily_returns) / 252
    if years <= 0:
        return 0.0
    return float(total ** (1 / years) - 1)


def sharpe_ratio(daily_returns: pd.Series) -> float:
    std = daily_returns.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float((daily_returns.mean() / std) * np.sqrt(252))


def sortino_ratio(daily_returns: pd.Series) -> float:
    downside = daily_returns[daily_returns < 0]
    dstd = downside.std(ddof=1)
    if dstd == 0 or np.isnan(dstd):
        return 0.0
    return float((daily_returns.mean() / dstd) * np.sqrt(252))


def max_drawdown(daily_returns: pd.Series) -> float:
    eq = (1 + daily_returns).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())


def calmar_ratio(daily_returns: pd.Series) -> float:
    mdd = abs(max_drawdown(daily_returns))
    if mdd == 0:
        return 0.0
    return annualized_return(daily_returns) / mdd


def compute_metrics(name: str, returns: pd.Series, turnover: float, trades: int) -> Dict:
    return {
        "strategy": name,
        "n_days": int(len(returns)),
        "cum_return": float((1 + returns).prod() - 1),
        "ann_return": annualized_return(returns),
        "ann_vol": float(returns.std(ddof=1) * np.sqrt(252)),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns),
        "avg_daily_return": float(returns.mean()),
        "turnover": float(turnover),
        "n_rebalances": int(trades),
    }


def to_feature_table(feat_panel: pd.DataFrame, date: pd.Timestamp, tickers: List[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        row = {"ticker": t}
        ok = True
        for f in ["ret_1d", "ret_5d", "ret_21d", "vol_21d", "mom_63d", "dd_63d"]:
            v = feat_panel.loc[date, (f, t)] if (f, t) in feat_panel.columns else np.nan
            row[f] = float(v) if pd.notna(v) else np.nan
            ok = ok and pd.notna(v)
        row["valid"] = ok
        rows.append(row)
    table = pd.DataFrame(rows)
    table = table[table["valid"]].drop(columns=["valid"]).reset_index(drop=True)
    return table


def clean_weights(raw: Dict[str, float], tickers: List[str], max_weight: float) -> Dict[str, float]:
    out = {t: 0.0 for t in tickers}
    for k, v in raw.items():
        if k in out:
            try:
                out[k] = max(0.0, min(float(v), max_weight))
            except Exception:
                out[k] = 0.0
    s = sum(out.values())
    if s <= 0:
        ew = 1.0 / len(tickers)
        return {t: ew for t in tickers}
    return {k: v / s for k, v in out.items()}


def cache_key(payload: Dict) -> str:
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_cache(path: str) -> Dict:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_cache(path: str, cache: Dict) -> None:
    Path(path).write_text(json.dumps(cache, indent=2))


def llm_allocate(
    client: OpenAI,
    model: str,
    temperature: float,
    feature_table: pd.DataFrame,
    prev_weights: Dict[str, float],
    horizon_days: int,
    position_aware: bool,
    max_weight: float,
    cache: Dict,
) -> Tuple[Dict[str, float], Dict]:
    tickers = feature_table["ticker"].tolist()
    features_json = feature_table.to_dict(orient="records")
    payload = {
        "model": model,
        "temperature": temperature,
        "horizon_days": horizon_days,
        "position_aware": position_aware,
        "features": features_json,
        "prev_weights": prev_weights if position_aware else {},
        "max_weight": max_weight,
    }
    key = cache_key(payload)
    if key in cache:
        return clean_weights(cache[key]["weights"], tickers, max_weight), {"cached": True, "usage": cache[key].get("usage", {})}

    style = "Use current holdings to reduce unnecessary churn." if position_aware else "Ignore any previous holdings and decide only from current features."
    user_prompt = {
        "task": "Allocate a long-only portfolio for the next horizon.",
        "horizon_days": horizon_days,
        "constraints": {
            "long_only": True,
            "sum_to_one": True,
            "max_weight_per_asset": max_weight,
            "prefer_top_k": 5,
        },
        "guidance": [
            "Favor assets with stronger medium-term momentum and controlled volatility.",
            "Penalize deep drawdowns and unstable short-term reversals.",
            style,
        ],
        "prev_weights": prev_weights if position_aware else {},
        "assets": features_json,
        "output_format": {"weights": {"TICKER": "float between 0 and max_weight"}, "rationale": "short string"},
    }

    retries = 5
    last_err = None
    for i in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a disciplined quantitative portfolio allocator. "
                            "Return strict JSON only with key 'weights' mapping tickers to numeric weights and key 'rationale'."
                        ),
                    },
                    {"role": "user", "content": json.dumps(user_prompt)},
                ],
                max_tokens=700,
            )
            text = resp.choices[0].message.content
            parsed = json.loads(text)
            weights = parsed.get("weights", {})
            cleaned = clean_weights(weights, tickers, max_weight)
            usage = {}
            if getattr(resp, "usage", None):
                usage = {
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                    "total_tokens": getattr(resp.usage, "total_tokens", None),
                }
            cache[key] = {"weights": cleaned, "rationale": parsed.get("rationale", ""), "usage": usage}
            return cleaned, {"cached": False, "usage": usage}
        except Exception as e:
            last_err = str(e)
            time.sleep(1.5 * (i + 1))

    raise RuntimeError(f"LLM call failed after retries: {last_err}")


def run_rebalance_strategy(
    name: str,
    cadence: int,
    dates: pd.DatetimeIndex,
    rets: pd.DataFrame,
    feat_panel: pd.DataFrame,
    tickers: List[str],
    mode: str,
    llm_client: OpenAI,
    cfg: Config,
    cache: Dict,
    position_aware: bool = True,
) -> Tuple[pd.Series, pd.DataFrame, Dict]:
    decision_dates = dates[::cadence]
    weights = {t: 1.0 / len(tickers) for t in tickers}
    daily = []
    trades = []
    total_turnover = 0.0
    api_calls = 0
    token_total = 0

    print(f"[run] {name}: {len(decision_dates)} rebalances (cadence={cadence}, mode={mode}, posaware={position_aware})")
    for i, d in enumerate(decision_dates):
        next_d = decision_dates[i + 1] if i + 1 < len(decision_dates) else dates[-1] + pd.Timedelta(days=1)
        hold_dates = dates[(dates >= d) & (dates < next_d)]
        if len(hold_dates) == 0:
            continue

        if i % 50 == 0:
            print(f"[run] {name}: step {i+1}/{len(decision_dates)} @ {d.date()}")

        if mode == "equal_weight":
            new_w = {t: 1.0 / len(tickers) for t in tickers}
        elif mode == "momentum":
            ft = to_feature_table(feat_panel, d, tickers)
            if ft.empty:
                new_w = {t: 1.0 / len(tickers) for t in tickers}
            else:
                top = ft.sort_values("mom_63d", ascending=False).head(cfg.top_k_hint)["ticker"].tolist()
                new_w = {t: (1.0 / len(top) if t in top else 0.0) for t in tickers}
        elif mode == "inv_vol":
            ft = to_feature_table(feat_panel, d, tickers)
            if ft.empty:
                new_w = {t: 1.0 / len(tickers) for t in tickers}
            else:
                tmp = {}
                for _, r in ft.iterrows():
                    v = max(r["vol_21d"], 1e-6)
                    tmp[r["ticker"]] = 1.0 / v
                s = sum(tmp.values())
                new_w = {t: tmp.get(t, 0.0) / s for t in tickers}
                new_w = clean_weights(new_w, tickers, cfg.max_weight)
        elif mode == "llm":
            ft = to_feature_table(feat_panel, d, tickers)
            if ft.empty:
                new_w = {t: 1.0 / len(tickers) for t in tickers}
            else:
                new_w, meta = llm_allocate(
                    client=llm_client,
                    model=cfg.model,
                    temperature=cfg.temperature,
                    feature_table=ft,
                    prev_weights=weights,
                    horizon_days=cadence,
                    position_aware=position_aware,
                    max_weight=cfg.max_weight,
                    cache=cache,
                )
                if not meta.get("cached", False):
                    api_calls += 1
                token_total += int(meta.get("usage", {}).get("total_tokens") or 0)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        turnover = sum(abs(new_w[t] - weights.get(t, 0.0)) for t in tickers)
        total_turnover += turnover
        cost = (cfg.tc_bps / 10000.0) * turnover

        for j, hd in enumerate(hold_dates):
            rr = float(sum(new_w[t] * rets.loc[hd, t] for t in tickers))
            if j == 0:
                rr -= cost
            daily.append({"date": hd, "strategy": name, "ret": rr})

        trades.append({"date": str(d.date()), "strategy": name, "turnover": turnover, "cost": cost, "weights": new_w})
        weights = new_w

    out = pd.DataFrame(daily).set_index("date").sort_index()["ret"]
    trades_df = pd.DataFrame(trades)
    aux = {"api_calls": api_calls, "token_total": token_total, "n_rebalances": int(len(decision_dates))}
    return out, trades_df, aux


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    more = 0
    less = 0
    for xi in x:
        more += np.sum(xi > y)
        less += np.sum(xi < y)
    n = len(x) * len(y)
    if n == 0:
        return 0.0
    return float((more - less) / n)


def circular_block_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int, block: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(a)
    if n == 0:
        return 0.0, 0.0

    diffs = []
    for _ in range(n_boot):
        idx = []
        while len(idx) < n:
            s = int(rng.integers(0, n))
            idx.extend([(s + k) % n for k in range(block)])
        idx = np.array(idx[:n])
        sa = pd.Series(a[idx])
        sb = pd.Series(b[idx])
        diffs.append(sortino_ratio(sa) - sortino_ratio(sb))

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def analyze(returns_df: pd.DataFrame, metrics_df: pd.DataFrame, cfg: Config) -> Dict:
    pivot = returns_df.pivot(index="date", columns="strategy", values="ret").sort_index()
    tests = {}

    compare_pairs = [
        ("llm_weekly_posaware", "llm_daily_posaware"),
        ("llm_monthly_posaware", "llm_daily_posaware"),
        ("llm_weekly_posaware", "llm_weekly_memoryless"),
    ]

    for a, b in compare_pairs:
        if a not in pivot.columns or b not in pivot.columns:
            continue
        aa = pivot[a].dropna()
        bb = pivot[b].dropna()
        common = aa.index.intersection(bb.index)
        x = aa.loc[common].values
        y = bb.loc[common].values
        diff = x - y

        p_shapiro = float(shapiro(diff).pvalue) if len(diff) >= 3 and len(diff) <= 5000 else math.nan
        try:
            w = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox", correction=True)
            p_wilcoxon = float(w.pvalue)
            w_stat = float(w.statistic)
        except Exception:
            p_wilcoxon = math.nan
            w_stat = math.nan

        ci_lo, ci_hi = circular_block_bootstrap_diff(x, y, cfg.n_bootstrap, cfg.bootstrap_block, cfg.seed)
        effect = cliffs_delta(x, y)

        tests[f"{a}_vs_{b}"] = {
            "n_common_days": int(len(common)),
            "mean_daily_diff": float(np.mean(diff)),
            "shapiro_p": p_shapiro,
            "wilcoxon_stat": w_stat,
            "wilcoxon_p": p_wilcoxon,
            "sortino_diff_bootstrap_ci95": [ci_lo, ci_hi],
            "cliffs_delta": effect,
        }

    best = metrics_df.sort_values("sortino", ascending=False).iloc[0]["strategy"]
    tests["best_strategy_by_sortino"] = str(best)
    return tests


def make_plots(returns_df: pd.DataFrame, cfg: Config) -> None:
    sns.set_style("whitegrid")
    pivot = returns_df.pivot(index="date", columns="strategy", values="ret").sort_index()

    equity = (1 + pivot.fillna(0.0)).cumprod()
    plt.figure(figsize=(11, 6))
    for c in equity.columns:
        plt.plot(equity.index, equity[c], label=c)
    plt.title("Equity Curves (Test Period)")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(Path(cfg.figures_dir) / "equity_curves.png", dpi=180)
    plt.close()

    drawdown = equity / equity.cummax() - 1
    plt.figure(figsize=(11, 6))
    for c in drawdown.columns:
        plt.plot(drawdown.index, drawdown[c], label=c)
    plt.title("Drawdown Curves")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(Path(cfg.figures_dir) / "drawdown_curves.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 6))
    sns.boxplot(data=pivot)
    plt.title("Daily Return Distribution by Strategy")
    plt.ylabel("Daily return")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(Path(cfg.figures_dir) / "return_boxplot.png", dpi=180)
    plt.close()


def run(cfg: Config) -> None:
    set_seed(cfg.seed)
    ensure_dirs(cfg)

    env = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "python": sys.version,
        "platform": sys.platform,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "model": cfg.model,
        "gpu_query": os.popen("nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo NO_GPU").read().strip(),
    }
    Path(cfg.output_dir, "environment.json").write_text(json.dumps(env, indent=2))
    Path(cfg.output_dir, "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    df = load_prices(cfg.price_path)
    quality_raw = data_quality_checks(df)
    df = feature_engineering(df)
    quality_feat = data_quality_checks(df)

    feat_panel, _, rets = prepare_panels(df)

    test_mask = (rets.index >= pd.to_datetime(cfg.test_start)) & (rets.index <= pd.to_datetime(cfg.test_end))
    dates = rets.index[test_mask]
    if len(dates) < 120:
        raise RuntimeError("Test period too short; need at least 120 trading days.")

    tickers = sorted(rets.columns.tolist())

    cache = load_cache(cfg.cache_path)
    client = build_client()

    strategies = [
        ("equal_weight", 21, "equal_weight", True),
        ("momentum_weekly", 5, "momentum", True),
        ("invvol_weekly", 5, "inv_vol", True),
        ("llm_daily_posaware", 1, "llm", True),
        ("llm_weekly_posaware", 5, "llm", True),
        ("llm_monthly_posaware", 21, "llm", True),
        ("llm_weekly_memoryless", 5, "llm", False),
    ]

    returns_all = []
    metrics_all = []
    trades_all = []
    usage_all = {}

    for name, cadence, mode, posaware in strategies:
        ret_s, trades_df, aux = run_rebalance_strategy(
            name=name,
            cadence=cadence,
            dates=dates,
            rets=rets,
            feat_panel=feat_panel,
            tickers=tickers,
            mode=mode,
            llm_client=client,
            cfg=cfg,
            cache=cache,
            position_aware=posaware,
        )
        returns_all.append(pd.DataFrame({"date": ret_s.index, "strategy": name, "ret": ret_s.values}))
        metrics_all.append(compute_metrics(name, ret_s, float(trades_df["turnover"].sum()), int(len(trades_df))))
        if not trades_df.empty:
            trades_all.append(trades_df)
        usage_all[name] = aux

    save_cache(cfg.cache_path, cache)

    returns_df = pd.concat(returns_all, ignore_index=True)
    metrics_df = pd.DataFrame(metrics_all).sort_values("sortino", ascending=False)
    trades_df = pd.concat(trades_all, ignore_index=True) if trades_all else pd.DataFrame()

    tests = analyze(returns_df, metrics_df, cfg)

    returns_df.to_csv(Path(cfg.output_dir) / "daily_returns.csv", index=False)
    metrics_df.to_csv(Path(cfg.output_dir) / "metrics.csv", index=False)
    Path(cfg.output_dir, "metrics.json").write_text(metrics_df.to_json(orient="records", indent=2))
    Path(cfg.output_dir, "stat_tests.json").write_text(json.dumps(tests, indent=2))
    Path(cfg.output_dir, "data_quality.json").write_text(json.dumps({"raw": quality_raw, "featured": quality_feat}, indent=2))
    Path(cfg.output_dir, "usage_summary.json").write_text(json.dumps(usage_all, indent=2))
    if not trades_df.empty:
        trades_df.to_json(Path(cfg.output_dir) / "trades.json", orient="records", indent=2)

    make_plots(returns_df, cfg)

    summary = {
        "best_by_sortino": tests.get("best_strategy_by_sortino"),
        "strategies": metrics_df[["strategy", "cum_return", "sortino", "max_drawdown", "turnover"]].to_dict(orient="records"),
    }
    Path(cfg.output_dir, "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4.1")
    parser.add_argument("--test-start", type=str, default="2025-01-02")
    parser.add_argument("--test-end", type=str, default="2026-01-30")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tc-bps", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config(
        model=args.model,
        test_start=args.test_start,
        test_end=args.test_end,
        temperature=args.temperature,
        tc_bps=args.tc_bps,
        seed=args.seed,
    )
    run(cfg)
