"""Core research utilities for the ML_Fin_Notebooks project.

Why this module exists
----------------------
The notebooks in this repository answer one question repeatedly: *is an observed
performance number real, or is it an artefact of how the experiment was run?*
Everything here exists to make that question answerable in code rather than in
a paragraph of prose.

The five invariants the whole project obeys
------------------------------------------
1. **Fit on the training window only.** Scalers, imputers, feature selectors and
   model hyper-parameters are fitted inside the fold. ``FoldSafePreprocessor``
   is the only supported way to transform features (see ``scaler`` helpers).
2. **Labels respect decision time.** ``make_direction_target`` documents the
   exact bar at which a label becomes observable.
3. **No same-bar execution.** :func:`positions_to_returns` applies a position to
   returns with a one-bar lag, so a signal can never earn money on the bar that
   produced it.
4. **Costs and a baseline are mandatory.** :func:`performance_report` always
   reports transaction costs and is always read next to buy-and-hold.
5. **Every number carries an error bar.** :func:`bootstrap_sharpe_ci`,
   :func:`probabilistic_sharpe_ratio`, :func:`deflated_sharpe_ratio` and
   :func:`min_track_record_length` are attached to reported metrics.

Conventions
-----------
* Prices are pandas ``Series`` indexed by a sorted, de-duplicated DatetimeIndex.
* All cross-validation helpers return **positional** ``numpy`` index arrays.
  Positional indices remove the single most common source of silent bugs in
  time-series code (index-label vs. integer-position confusion).
* Annualisation assumes 252 trading days unless stated otherwise.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "TRADING_DAYS",
    "DataSource",
    "synthetic_ohlcv",
    "sanitize_ohlcv",
    "load_ohlcv",
    "technical_indicators",
    "make_feature_frame",
    "make_direction_target",
    "make_regression_target",
    "FoldSafePreprocessor",
    "apply_causal_execution",
    "positions_to_returns",
    "buy_and_hold_returns",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "annualised_return",
    "profit_factor",
    "win_rate",
    "performance_report",
    "block_bootstrap_indices",
    "bootstrap_sharpe_ci",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "min_track_record_length",
    "walk_forward_splits",
    "purged_kfold_splits",
    "combinatorial_purged_cv_splits",
    "cpcv_backtest_paths",
    "reliability_table",
    "expected_calibration_error",
    "brier_score",
    "brier_skill_score",
    "split_conformal_interval",
    "conformal_coverage",
]

TRADING_DAYS = 252
_EULER_MASCHERONI = 0.5772156649015329


# --------------------------------------------------------------------------- #
# Data acquisition
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DataSource:
    """Provenance of a price series. Printed at the top of every notebook."""

    name: str
    kind: str  # "local" | "yfinance" | "synthetic"
    note: str = ""

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.name} ({self.kind}{': ' + self.note if self.note else ''})"


def synthetic_ohlcv(
    n_days: int = 1500,
    start: str = "2019-01-02",
    seed: int = 7,
    annual_drift: float = 0.12,
    annual_vol: float = 0.55,
    ticker: str = "SYNTH",
) -> pd.DataFrame:
    """Deterministic synthetic OHLCV series with fat tails and volatility clustering.

    Used only as an offline fallback so the notebooks remain runnable without
    network access. The generator is a two-state regime-switching model so the
    synthetic series has properties real financial series have (fat tails,
    clustered volatility) instead of being benign Gaussian noise -- a naive
    Gaussian series would make ARIMA diagnostics look artificially good.

    The output is explicitly labelled ``SYNTH`` so synthetic data can never be
    mistaken for a market observation in a printed result table.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)
    dt = 1.0 / TRADING_DAYS

    # Two-regime Markov switching between calm and stressed volatility.
    p_stay = 0.985
    regime = np.zeros(n_days, dtype=int)
    for i in range(1, n_days):
        stay = rng.random() < p_stay
        regime[i] = regime[i - 1] if stay else 1 - regime[i - 1]

    base_vol = annual_vol * np.sqrt(dt)
    vol = base_vol * np.where(regime == 1, 2.4, 1.0)

    # Student-t innovations -> fat tails.
    nu = 4.0
    shocks = rng.standard_t(df=nu, size=n_days) / np.sqrt(nu / (nu - 2.0))
    log_ret = annual_drift * dt - 0.5 * vol**2 + vol * shocks

    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = close / np.exp(log_ret * 0.5)
    intrabar = np.abs(rng.normal(0.0, 0.6, size=n_days)) * vol * close
    high = np.maximum(open_, close) + intrabar
    low = np.minimum(open_, close) - intrabar
    volume = rng.lognormal(mean=15.5, sigma=0.7, size=n_days)

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=idx,
    ).rename_axis("Date")


def sanitize_ohlcv(df: pd.DataFrame, price_columns: Sequence[str] = ("Open", "High", "Low", "Close")) -> pd.DataFrame:
    """Return a clean OHLCV frame: sorted, de-duplicated, numeric, no NaNs.

    Note the asymmetry that matters for leakage: gaps are filled **forward
    only**. ``bfill`` would copy a future price into a past row and silently
    create look-ahead information.
    """
    if df is None or len(df) == 0:
        raise ValueError("empty price frame")

    out = df.copy()
    out.index = pd.to_datetime(out.index)
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out = out.loc[:, ~out.columns.duplicated()]
    out.index = out.index.normalize()
    out = out[~out.index.duplicated(keep="first")].sort_index()

    for col in price_columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=[c for c in ("Low", "Close") if c in out.columns])

    full = pd.date_range(out.index.min(), out.index.max(), freq="B")
    out = out.reindex(full).ffill()
    return out.dropna(subset=[c for c in ("Low", "Close") if c in out.columns])


def load_ohlcv(
    ticker: str,
    start: str = "2019-01-01",
    end: str | None = None,
    local_dir: str | None = None,
    allow_synthetic: bool = True,
    synthetic_kwargs: dict | None = None,
) -> tuple[pd.DataFrame, DataSource]:
    """Load OHLCV with an explicit, auditable fallback chain.

    Order: local CSV cache -> Yahoo Finance -> synthetic. The returned
    :class:`DataSource` must be displayed in the notebook so a reader always
    knows whether a printed number came from a market or from a simulator.
    """
    import os

    end = end or pd.Timestamp.utcnow().strftime("%Y-%m-%d")

    if local_dir:
        path = os.path.join(local_dir, f"{ticker.replace('-', '_')}.csv")
        if os.path.exists(path):
            raw = pd.read_csv(path, index_col=0, parse_dates=True)
            return sanitize_ohlcv(raw), DataSource(ticker, "local", path)

    try:
        import yfinance as yf

        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if raw is not None and not raw.empty:
            return sanitize_ohlcv(raw), DataSource(ticker, "yfinance", f"{start}..{end}")
        raise ValueError("empty frame from yfinance")
    except Exception as exc:  # network, rate limit, delisted ticker
        if not allow_synthetic:
            raise
        kwargs = dict(n_days=1500, seed=abs(hash(ticker)) % (2**31), ticker=ticker)
        kwargs.update(synthetic_kwargs or {})
        return synthetic_ohlcv(**kwargs), DataSource(ticker, "synthetic", f"fallback after {type(exc).__name__}")


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    return pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Classic indicators implemented in pandas/numpy only.

    TA-Lib and pandas-ta are deliberately avoided: keeping the numerics
    dependency-free makes every notebook runnable from a bare pandas+numpy
    environment, and makes the formulas auditable in one place.
    """
    out = df.copy()
    close = out["Close"]

    out["ret_1"] = close.pct_change()
    out["ema_fast"] = _ema(close, 12)
    out["ema_slow"] = _ema(close, 26)
    out["macd"] = out["ema_fast"] - out["ema_slow"]
    out["macd_signal"] = _ema(out["macd"], 9)
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    out["rsi_14"] = _rsi(close, 14)
    out["atr_14"] = _true_range(out).ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
    out["vol_20"] = out["ret_1"].rolling(20, min_periods=5).std()
    out["ma_ratio"] = close / close.rolling(20, min_periods=5).mean() - 1.0
    return out


def make_feature_frame(
    df: pd.DataFrame,
    n_lags: int = 5,
    extra_indicators: bool = True,
) -> pd.DataFrame:
    """Build a strictly causal feature matrix.

    Every column is a function of information available at the close of bar
    ``t``. Lags are explicit (``Close_Lag_1`` is yesterday's close), rolling
    windows are trailing, and no column is centred. The one place where
    future information could enter -- the normalisation of the *close* used for
    reporting -- is handled by the caller, never here.
    """
    out = technical_indicators(df) if extra_indicators else df.copy()
    close = out["Close"]

    for lag in range(1, n_lags + 1):
        out[f"Close_Lag_{lag}"] = close.shift(lag)
        out[f"ret_lag_{lag}"] = out["ret_1"].shift(lag)
    out["vol_ratio"] = out["vol_20"] / out["vol_20"].rolling(60, min_periods=10).mean()
    out["range_pct"] = (out["High"] - out["Low"]) / close
    out["close_loc"] = (close - out["Low"]) / (out["High"] - out["Low"]).replace(0.0, np.nan)

    feature_cols = [c for c in out.columns if c not in ("Open", "High", "Low", "Close", "Volume")]
    out = out[feature_cols]
    # ffill only. bfill would import future prices into the past.
    return out.ffill()


def make_direction_target(
    close: pd.Series,
    horizon: int = 1,
    deadband: float = 0.0,
    labels: tuple[int, int, int] = (0, 1, 2),
) -> pd.Series:
    """Three-class direction label with an explicit decision-time contract.

    ``labels = (sell, buy, hold)``. A bar is labelled by the sign of the
    ``horizon``-bar forward return; a return inside ``+/- deadband`` is a
    *hold* class.

    Decision-time contract: the label for bar ``t`` becomes observable at the
    close of bar ``t + horizon``. Any model that consumes it must therefore be
    trained only on labels that closed before the first bar it is scored on.
    That offset is what :func:`~mlfin_core.walk_forward_splits` and
    :func:`~mlfin_core.purged_kfold_splits` enforce via their ``embargo`` /
    ``t1`` arguments -- a horizon of ``h`` days requires an embargo of at
    least ``h``.

    The final ``horizon`` bars are dropped because their label is not yet known.
    """
    fwd = close.shift(-horizon) / close - 1.0
    sell, buy, hold = labels
    target = pd.Series(hold, index=close.index, dtype="float64")
    target[fwd < -deadband] = sell
    target[fwd > deadband] = buy
    return target.iloc[:-horizon] if horizon > 0 else target


def make_regression_target(close: pd.Series, horizon: int = 1) -> pd.Series:
    """``horizon``-bar forward log return, with the unobservable tail dropped."""
    target = np.log(close.shift(-horizon) / close)
    return target.iloc[:-horizon] if horizon > 0 else target


# --------------------------------------------------------------------------- #
# Leak-free preprocessing
# --------------------------------------------------------------------------- #


class FoldSafePreprocessor:
    """Median imputation + standardisation fitted **inside** each training fold.

    This class is the reference implementation of invariant #1. The
    deliberate contrast is :meth:`fit_on_full_sample`, which reproduces the
    single most common time-series bug in applied ML -- fitting the scaler on
    the concatenation of train and test before splitting. HW7 measures how
    much that bug inflates reported performance.
    """

    def __init__(self, robust: bool = False) -> None:
        self.robust = robust
        self.columns_: list[str] | None = None
        self.center_: pd.Series | None = None
        self.scale_: pd.Series | None = None

    # -- fit ---------------------------------------------------------------- #
    def fit(self, X: pd.DataFrame) -> "FoldSafePreprocessor":
        self.columns_ = list(X.columns)
        med = X.median(numeric_only=True)
        self.center_ = med
        if self.robust:
            q1, q3 = X.quantile(0.25), X.quantile(0.75)
            scale = (q3 - q1).replace(0.0, np.nan)
        else:
            scale = X.std(ddof=0).replace(0.0, np.nan)
        self.scale_ = scale.fillna(1.0).replace(0.0, 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.columns_ is None:
            raise RuntimeError("FoldSafePreprocessor.transform called before fit")
        out = X.reindex(columns=self.columns_)
        out = out.fillna(self.center_).fillna(0.0)
        return (out - self.center_) / self.scale_

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    # -- the anti-pattern ---------------------------------------------------- #
    @staticmethod
    def fit_on_full_sample(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Leaky variant: statistics are taken from train+test jointly.

        Kept only so HW7 can quantify the damage. Never use in production.
        """
        combined = pd.concat([X_train, X_test], axis=0)
        leaky = FoldSafePreprocessor().fit(combined)
        n_train = len(X_train)
        return leaky.transform(combined).iloc[:n_train], leaky.transform(combined).iloc[n_train:]


# --------------------------------------------------------------------------- #
# Execution, costs and metrics
# --------------------------------------------------------------------------- #


def apply_causal_execution(
    probabilities: np.ndarray,
    buy: int = 1,
    sell: int = 0,
    confirm_bars: int = 2,
    min_confidence: float = 0.40,
) -> np.ndarray:
    """Turn class probabilities into a discrete position with two safeguards.

    A class prediction becomes a *desired* position, then two rules apply:

    1. **Abstention.** If the winning class holds less than ``min_confidence``
       of the probability mass, the desired position is flat (stay in cash).
       A model is allowed to decline to trade. The default of 0.40 assumes a
       3-class problem: the chance level is ``1/3``, so anything at or below it
       would never trigger, and 0.40 demands a real edge over the prior.
    2. **Reversal confirmation.** Changing the live position requires
       ``confirm_bars`` *consecutive* bars wanting the same new position.
       Without this, a classifier oscillating around 0.5 churns the book and
       pays the spread on every flip.

    The returned position for bar ``t`` is decided using bar ``t`` information
    only; :func:`positions_to_returns` applies the additional one-bar execution
    lag.
    """
    probs = np.asarray(probabilities, dtype=float)
    best = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    desired = np.where(best == buy, 1.0, np.where(best == sell, -1.0, 0.0))
    desired = np.where(confidence < min_confidence, 0.0, desired)

    signal = np.zeros(len(desired), dtype=float)
    if len(desired) == 0:
        return signal

    current = desired[0]
    pending: float | None = None
    run = 0
    for t in range(len(desired)):
        d = float(desired[t])
        if d == current:
            pending, run = None, 0
        else:
            if d == pending:
                run += 1
            else:
                pending, run = d, 1
            if run >= confirm_bars:
                current, pending, run = d, None, 0
        signal[t] = current
    return signal


def positions_to_returns(
    positions: np.ndarray | pd.Series,
    prices: pd.Series,
    cost_bps: float = 10.0,
    lag: int = 1,
) -> pd.Series:
    """Daily strategy returns from a position series, net of transaction costs.

    ``lag=1`` implements invariant #3: the position decided on bar ``t-1`` earns
    the return of bar ``t``. Cost is charged on every change of position,
    expressed in basis points of notional.
    """
    pos = pd.Series(np.asarray(positions, dtype=float), index=prices.index).reindex(prices.index).fillna(0.0)
    asset_ret = prices.pct_change().fillna(0.0)
    held = pos.shift(lag).fillna(0.0)
    turnover = held.diff().abs().fillna(held.abs())
    cost = turnover * (cost_bps / 10_000.0)
    return (held * asset_ret - cost).rename("strategy_return")


def buy_and_hold_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().fillna(0.0).rename("buy_and_hold_return")


def sharpe_ratio(returns: pd.Series, rf_annual: float = 0.0) -> float:
    """Annualised Sharpe ratio; ``nan`` when the return stream has no dispersion.

    The dispersion guard uses an absolute tolerance because a *constant*
    return series does not produce an exactly zero standard deviation in
    floating point, and dividing by ~1e-19 would report a Sharpe of 1e16
    instead of admitting that the ratio is undefined.
    """
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if not np.isfinite(sd) or sd <= 1e-12:
        return float("nan")
    return float((r.mean() - rf_annual / TRADING_DAYS) / sd * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: pd.Series, rf_annual: float = 0.0, mar_annual: float = 0.0) -> float:
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return float("nan")
    downside = r[r < 0.0]
    dd = downside.std(ddof=1)
    if not np.isfinite(dd) or dd <= 1e-12:
        return float("nan")
    return float((r.mean() - (mar_annual + rf_annual) / TRADING_DAYS) / dd * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + pd.Series(returns).fillna(0.0)).cumprod()
    peak = equity.cummax()
    return float(((equity - peak) / peak).min())


def annualised_return(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return float("nan")
    total = float((1.0 + r).prod())
    if total <= 0:
        return -1.0
    return float(total ** (periods / len(r)) - 1.0)


def profit_factor(returns: pd.Series) -> float:
    r = pd.Series(returns).dropna()
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def win_rate(returns: pd.Series) -> float:
    r = pd.Series(returns).dropna()
    if len(r) == 0:
        return float("nan")
    return float((r > 0).mean())


def performance_report(returns: pd.Series, costs_bps: float = 10.0) -> dict:
    """Single-call metric bundle. Always read ``sharpe`` next to its error bar."""
    r = pd.Series(returns).dropna()
    return {
        "n_obs": int(len(r)),
        "total_return": float((1.0 + r).prod() - 1.0) if len(r) else float("nan"),
        "annualised_return": annualised_return(r),
        "sharpe": sharpe_ratio(r),
        "sortino": sortino_ratio(r),
        "max_drawdown": max_drawdown(r),
        "win_rate": win_rate(r),
        "profit_factor": profit_factor(r),
        "daily_vol": float(r.std(ddof=1)) if len(r) > 1 else float("nan"),
        "costs_bps": costs_bps,
    }


# --------------------------------------------------------------------------- #
# Uncertainty
# --------------------------------------------------------------------------- #


def block_bootstrap_indices(
    n: int,
    block_size: int,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Circular block bootstrap index matrix of shape ``(n_samples, n)``.

    A block bootstrap is required because daily strategy returns are strongly
    autocorrelated in the volatility dimension; an i.i.d. bootstrap would
    produce confidence intervals that are far too narrow and would routinely
    declare noise as edge.
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    if n == 0:
        raise ValueError("n must be > 0")
    block_size = min(block_size, n)
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=(n_samples, n_blocks))
    offsets = np.arange(block_size)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    return idx.reshape(n_samples, -1)[:, :n]


def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_boot: int = 2000,
    block_size: int = 10,
    alpha: float = 0.05,
    seed: int = 0,
    rf_annual: float = 0.0,
) -> dict:
    """Block-bootstrap confidence interval for the annualised Sharpe ratio."""
    r = pd.Series(returns).dropna().to_numpy()
    point = sharpe_ratio(pd.Series(r), rf_annual=rf_annual)
    if len(r) < max(30, block_size * 3):
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "p_gt_zero": float("nan"), "n_boot": 0}

    rng = np.random.default_rng(seed)
    idx = block_bootstrap_indices(len(r), block_size, n_boot, rng)
    samples = r[idx]
    rf = rf_annual / TRADING_DAYS
    mean = samples.mean(axis=1)
    std = samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        boots = np.where(std > 0, (mean - rf) / std * np.sqrt(TRADING_DAYS), np.nan)
    boots = boots[np.isfinite(boots)]
    if len(boots) == 0:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "p_gt_zero": float("nan"), "n_boot": 0}

    return {
        "point": point,
        "lo": float(np.quantile(boots, alpha / 2)),
        "hi": float(np.quantile(boots, 1 - alpha / 2)),
        "p_gt_zero": float((boots > 0).mean()),
        "n_boot": int(len(boots)),
    }


def _higher_moments(x: np.ndarray) -> tuple[float, float, float]:
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    s, k = stats.skew(x, bias=False), stats.kurtosis(x, fisher=False, bias=False)
    return float(s), float(k), float(n)


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    sr_benchmark: float = 0.0,
    rf_annual: float = 0.0,
) -> float:
    """PSR: probability that true SR exceeds ``sr_benchmark`` (Bailey & Lopez de Prado, 2014).

    The denominator carries the skewness and kurtosis of the realised return
    stream, so a non-normal stream needs materially more observations to reach
    the same confidence than a Gaussian one.
    """
    r = pd.Series(returns).dropna().to_numpy()
    sr = sharpe_ratio(pd.Series(r), rf_annual=rf_annual) / np.sqrt(TRADING_DAYS)
    skew, kurt, n = _higher_moments(r)
    if not np.isfinite(sr) or not np.isfinite(n) or n < 3 or sr == 0:
        return float("nan")
    var_term = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if var_term <= 0:
        return float("nan")
    z = (sr - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(var_term)
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int = 1,
    rf_annual: float = 0.0,
) -> dict:
    """DSR: PSR against the Sharpe expected from the best of ``n_trials`` trials.

    Run 50 hyper-parameter configurations and report the best? The best
    configuration's Sharpe is biased upward by selection. DSR subtracts the
    expected maximum of ``n_trials`` Gaussian Sharpes, which is the honest
    benchmark to test against. Passing ``n_trials`` equal to the number of
    configurations actually tried is the whole point.
    """
    r = pd.Series(returns).dropna().to_numpy()
    psr = probabilistic_sharpe_ratio(r, sr_benchmark=0.0, rf_annual=rf_annual)
    sr = sharpe_ratio(pd.Series(r), rf_annual=rf_annual) / np.sqrt(TRADING_DAYS)
    skew, kurt, n = _higher_moments(r)
    if not np.isfinite(n) or n < 3:
        return {"sr": sr, "sr_benchmark": float("nan"), "psr": psr, "dsr": float("nan")}

    # Expected maximum of n_trials iid standard normals. With s ~ SR * sqrt(n-1)
    # the per-trial standard deviation, the expected max is
    #   s * [(1-g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N * e))]
    # where g is the Euler-Mascheroni constant and e its exponential.
    if n_trials > 1:
        g = _EULER_MASCHERONI
        e = float(np.exp(g))
        n_ = float(n_trials)
        term_a = (1.0 - g) * stats.norm.ppf(1.0 - 1.0 / n_)
        term_b = g * stats.norm.ppf(1.0 - 1.0 / (n_ * e))
        sr0 = (term_a + term_b) / np.sqrt(n - 1.0)
    else:
        sr0 = 0.0

    return {
        "sr": float(sr),
        "sr_benchmark": float(sr0),
        "psr": psr,
        "dsr": probabilistic_sharpe_ratio(r, sr_benchmark=sr0, rf_annual=rf_annual),
    }


def min_track_record_length(
    returns: pd.Series,
    sr_target: float = 0.0,
    rf_annual: float = 0.0,
) -> float:
    """MinTRL: observations needed for SR evidence to beat ``sr_target``.

    A backtest that "works" over 6 trades does not have enough observations to
    establish that its Sharpe is non-zero, however attractive the number is.
    """
    r = pd.Series(returns).dropna().to_numpy()
    sr = sharpe_ratio(pd.Series(r), rf_annual=rf_annual) / np.sqrt(TRADING_DAYS)
    skew, kurt, _ = _higher_moments(r)
    if not np.isfinite(sr) or sr <= 0:
        return float("nan")
    var_term = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if var_term <= 0:
        return float("nan")
    return float(1.0 + var_term * (sr_target / sr) ** 2)


# --------------------------------------------------------------------------- #
# Cross-validation schemes
# --------------------------------------------------------------------------- #


def _validate_bounds(n_obs: int, n_splits: int, min_train_size: int, test_size: int, embargo: int) -> None:
    if n_obs <= 0:
        raise ValueError("n_obs must be positive")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if embargo < 0:
        raise ValueError("embargo must be >= 0")
    if min_train_size + n_splits * (test_size + embargo) > n_obs:
        raise ValueError(
            f"scheme does not fit: need > {min_train_size + n_splits * (test_size + embargo)} obs, got {n_obs}"
        )


def walk_forward_splits(
    n_obs: int,
    n_splits: int = 5,
    min_train_size: int | None = None,
    test_size: int | None = None,
    embargo: int = 0,
    expanding: bool = True,
    train_window: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Anchored walk-forward splits with an embargo gap.

    ``expanding=True`` grows the training window from the start of the series
    (the standard backtest protocol, maximum data, maximum overfitting risk).
    ``expanding=False`` slides a fixed ``train_window`` so that only recent
    regime behaviour is learned -- usually the more honest choice on
    non-stationary data, and worth reporting both of.

    The embargo removes ``embargo`` bars immediately *after* each test block
    from the next training window. Without it, a bar whose label closes inside
    the test block can still sit in the next training set, which re-injects the
    look-ahead the split was built to remove. Set ``embargo >= horizon``.
    """
    if min_train_size is None:
        min_train_size = max(60, n_obs // 3)
    if expanding:
        train_window = None
    elif train_window is None:
        train_window = min_train_size
    if test_size is None:
        test_size = max(1, (n_obs - min_train_size - n_splits * embargo) // n_splits)
    _validate_bounds(n_obs, n_splits, min_train_size, test_size, embargo)

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    cursor = min_train_size
    for _ in range(n_splits):
        test_idx = np.arange(cursor, cursor + test_size)
        if len(test_idx) == 0:
            break
        # Training ends `embargo` bars before the test block opens.
        train_end = cursor - embargo
        train_start = max(0, train_end - train_window) if train_window else 0
        train_idx = np.arange(train_start, max(train_end, 1))
        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))
        cursor += test_size + embargo
    return splits


def purged_kfold_splits(
    n_obs: int,
    n_splits: int = 5,
    embargo: int = 0,
    t1: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Purged K-fold with embargo (Lopez de Prado, *AFML*, ch. 7).

    Two mechanisms:

    * **Purge** -- when a contiguous block of ``t1 + 1`` bars is the test set,
      every training observation whose label window ``[t, t + t1]`` overlaps
      that block is dropped. This is what makes a K-fold legitimate for
      overlapping financial labels.
    * **Embargo** -- after the test block, the next ``embargo`` bars are
      dropped from the training set, because their labels are still
      contaminated by the test period.

    ``t1`` is the label horizon in bars.
    """
    if n_splits < 2 or n_splits > n_obs:
        raise ValueError("n_splits must be in [2, n_obs]")
    if t1 < 0 or embargo < 0:
        raise ValueError("t1 and embargo must be non-negative")

    bounds = np.linspace(0, n_obs, n_splits + 1).astype(int)
    all_idx = np.arange(n_obs)
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for i in range(n_splits):
        test_idx = all_idx[bounds[i] : bounds[i + 1]]
        if len(test_idx) == 0:
            continue
        test_start, test_end = int(test_idx[0]), int(test_idx[-1])
        test_window = np.zeros(n_obs, dtype=bool)
        test_window[test_start : test_end + 1] = True

        if t1 > 0:
            # Label of observation j spans [j, j + t1].
            label_end = all_idx + t1
            overlaps = test_window[np.clip(label_end, 0, n_obs - 1)]
            overlaps |= test_window[all_idx]
            train_mask = ~overlaps
        else:
            train_mask = ~test_window

        if embargo > 0:
            lo = test_end + 1
            hi = min(lo + embargo, n_obs)
            train_mask[lo:hi] = False

        train_idx = all_idx[train_mask]
        if len(train_idx) and len(test_idx):
            splits.append((train_idx, test_idx))
    return splits


def combinatorial_purged_cv_splits(
    n_obs: int,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo: int = 0,
    t1: int = 0,
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """CPCV: group the data, then return every combination of test groups.

    Each element of the outer list is one complete backtest path built from
    ``n_test_groups`` disjoint purged test blocks. The distribution of Sharpe
    across paths is a far better uncertainty estimate than a single path, and
    it answers the question a single hold-out cannot: *how much does my result
    depend on where the test period happened to start?*
    """
    if n_test_groups >= n_groups:
        raise ValueError("n_test_groups must be < n_groups")
    from itertools import combinations

    base = purged_kfold_splits(n_obs, n_splits=n_groups, embargo=embargo, t1=t1)
    if len(base) < n_groups:
        raise ValueError("not enough groups produced by purged_kfold_splits")

    paths: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for combo in combinations(range(n_groups), n_test_groups):
        path: list[tuple[np.ndarray, np.ndarray]] = []
        for g in combo:
            path.append(base[g])
        # Chronological order within a path.
        path.sort(key=lambda s: int(s[1][0]))
        paths.append(path)
    return paths


def cpcv_backtest_paths(
    oof_predictions: np.ndarray,
    paths: list[list[tuple[np.ndarray, np.ndarray]]],
) -> dict:
    """Assemble per-path equity curves from out-of-fold predictions.

    ``oof_predictions`` must be a *signal* series (e.g. a discrete position)
    already computed out-of-fold. The bars in a path's test blocks are
    concatenated chronologically; a bar never appears twice in a path because
    the test blocks are disjoint.
    """
    per_path: list[pd.Series] = []
    for path in paths:
        idx = np.concatenate([test for _, test in path])
        per_path.append(pd.Series(np.asarray(oof_predictions)[idx]))
    if not per_path:
        return {"n_paths": 0}
    length = min(len(p) for p in per_path)
    stacked = pd.DataFrame({f"path_{i}": p.iloc[:length].reset_index(drop=True) for i, p in enumerate(per_path)})
    return {
        "n_paths": len(per_path),
        "path_sharpe": {c: sharpe_ratio(stacked[c]) for c in stacked.columns},
        "mean_path_sharpe": float(np.mean([sharpe_ratio(stacked[c]) for c in stacked.columns])),
        "p05_path_sharpe": float(np.quantile([sharpe_ratio(stacked[c]) for c in stacked.columns], 0.05)),
        "p95_path_sharpe": float(np.quantile([sharpe_ratio(stacked[c]) for c in stacked.columns], 0.95)),
    }


# --------------------------------------------------------------------------- #
# Probabilistic forecasting and calibration
# --------------------------------------------------------------------------- #


def reliability_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Reliability diagram data: observed frequency vs mean predicted probability."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_id = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_id == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": b,
                "n": int(mask.sum()),
                "pred_mean": float(y_prob[mask].mean()),
                "obs_freq": float(y_true[mask].mean()),
                "gap": float(y_true[mask].mean() - y_prob[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """ECE with equal-width bins: sample-weighted mean |predicted - observed|."""
    table = reliability_table(y_true, y_prob, n_bins)
    if table.empty:
        return float("nan")
    return float((table["n"] * table["gap"].abs()).sum() / table["n"].sum())


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Multi-class Brier score via the standard one-hot decomposition."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_classes = y_prob.shape[1]
    onehot = np.zeros_like(y_prob)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((y_prob - onehot) ** 2, axis=1)))


def brier_skill_score(y_true: np.ndarray, y_prob: np.ndarray, climatology: np.ndarray | None = None) -> float:
    """BSS against a reference forecast; 1 - BSS is the skill score.

    A negative value means the model is worse than the reference -- the check
    that stops a fluent-looking forecast from being reported as skilful.
    """
    brier = brier_score(y_true, y_prob)
    if climatology is None:
        y_true = np.asarray(y_true, dtype=int)
        counts = np.bincount(y_true, minlength=np.asarray(y_prob).shape[1]).astype(float)
        clim = counts / counts.sum()
        climatology = np.tile(clim, (len(y_true), 1))
    ref = brier_score(y_true, climatology)
    if ref == 0:
        return float("nan")
    return float(1.0 - brier / ref)


def split_conformal_interval(
    y_calib: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.1,
) -> tuple[float, float]:
    """Split-conformal prediction interval around point forecasts.

    Nonconformity score is the absolute residual on a held-out *calibration*
    split. The interval is the empirical ``ceil((n+1)(1-alpha))/n`` quantile of
    those scores, which gives finite-sample coverage of at least ``1 - alpha``
    **without any distributional assumption** -- no normality, no
    homoscedasticity. That is the property that matters on financial returns,
    where both assumptions fail.

    Requires the calibration split to be disjoint from the fitting split and to
    precede the test period in time.
    """
    y_calib = np.asarray(y_calib, dtype=float).ravel()
    resid = np.abs(y_calib - np.asarray(y_pred, dtype=float).ravel())
    n = len(resid)
    if n == 0:
        raise ValueError("empty calibration set")
    q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    q = float(np.quantile(resid, q_level, method="higher"))
    return -q, q


def conformal_coverage(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict:
    """Empirical coverage of a conformal interval, plus mean width."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    inside = (y_true >= np.asarray(lower).ravel()) & (y_true <= np.asarray(upper).ravel())
    return {
        "empirical_coverage": float(inside.mean()),
        "mean_width": float(np.mean(np.asarray(upper).ravel() - np.asarray(lower).ravel())),
        "n": int(len(y_true)),
    }
