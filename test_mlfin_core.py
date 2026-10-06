"""Unit tests for :mod:`mlfin_core`.

These tests check the *invariants*, not the outputs. Each class targets a
specific way a time-series research pipeline can lie to its author:

* ``TestNoLookAhead`` -- asserts no function can see a future bar.
* ``TestSplits`` -- asserts cross-validation never leaks across a boundary.
* ``TestUncertainty`` -- asserts the error bars are calibrated, not decorative.
* ``TestConformal`` -- asserts finite-sample coverage holds without normality.

Run with ``python -m pytest test_mlfin_core.py -q`` from the repository root.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlfin_core import (
    FoldSafePreprocessor,
    apply_causal_execution,
    block_bootstrap_indices,
    bootstrap_sharpe_ci,
    brier_skill_score,
    buy_and_hold_returns,
    combinatorial_purged_cv_splits,
    conformal_coverage,
    deflated_sharpe_ratio,
    expected_calibration_error,
    make_direction_target,
    make_feature_frame,
    make_regression_target,
    max_drawdown,
    min_track_record_length,
    performance_report,
    positions_to_returns,
    probabilistic_sharpe_ratio,
    profit_factor,
    purged_kfold_splits,
    reliability_table,
    sanitize_ohlcv,
    sharpe_ratio,
    sortino_ratio,
    split_conformal_interval,
    synthetic_ohlcv,
    technical_indicators,
    walk_forward_splits,
)


@pytest.fixture(scope="module")
def prices() -> pd.Series:
    """Synthetic log-price series with drift and stochastic volatility."""
    rng = np.random.default_rng(11)
    n = 1200
    vol = 0.011 * (1.0 + 0.5 * (rng.random(n) < 0.08))
    log_ret = 0.0006 - 0.5 * vol**2 + vol * rng.standard_t(5, size=n)
    return pd.Series(100.0 * np.exp(np.cumsum(log_ret)), index=pd.bdate_range("2020-01-01", periods=n))


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    return synthetic_ohlcv(n_days=900, seed=3)


class TestData:
    def test_synthetic_is_deterministic(self) -> None:
        pd.testing.assert_frame_equal(synthetic_ohlcv(n_days=200, seed=42), synthetic_ohlcv(n_days=200, seed=42))

    def test_synthetic_seed_changes_path(self) -> None:
        a = synthetic_ohlcv(n_days=200, seed=1)
        b = synthetic_ohlcv(n_days=200, seed=2)
        assert not a["Close"].equals(b["Close"])

    def test_ohlc_invariants_hold(self, ohlcv: pd.DataFrame) -> None:
        assert (ohlcv["High"] >= ohlcv["Low"]).all()
        assert (ohlcv["High"] >= ohlcv["Close"]).all()
        assert (ohlcv["Low"] <= ohlcv["Close"]).all()
        assert (ohlcv["Volume"] > 0).all()

    def test_sanitize_deduplicates_and_sorts(self) -> None:
        raw = pd.DataFrame(
            {"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Volume": 10.0},
            index=pd.to_datetime(["2020-01-03", "2020-01-02", "2020-01-02", "2020-01-01"]),
        )
        out = sanitize_ohlcv(raw)
        assert out.index.is_monotonic_increasing
        assert out.index.is_unique
        assert not out.isna().any().any()

    def test_sanitize_flattens_multiindex_columns(self) -> None:
        cols = pd.MultiIndex.from_product([["Close", "Open"], ["SPY"]])
        raw = pd.DataFrame([[1.0, 1.0]], index=pd.to_datetime(["2020-01-01"]), columns=cols)
        out = sanitize_ohlcv(raw)
        assert isinstance(out.columns, pd.Index)
        assert "Close" in out.columns

    def test_sanitize_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            sanitize_ohlcv(pd.DataFrame())


class TestNoLookAhead:
    """A pipeline that can see the future fails these tests. That is the point."""

    def test_features_are_causal_under_truncation(self, ohlcv: pd.DataFrame) -> None:
        """Recomputing on a truncated history must reproduce the earlier rows.

        If any feature used future information, appending new bars would
        retroactively change earlier feature values -- and therefore earlier
        training labels -- which is exactly the bug.
        """
        full = make_feature_frame(ohlcv, n_lags=3)
        truncated = make_feature_frame(ohlcv.iloc[:700], n_lags=3)
        common = truncated.index.intersection(full.index)
        pd.testing.assert_frame_equal(
            full.loc[common].round(10), truncated.loc[common].round(10), check_names=False
        )

    def test_target_is_shifted_into_the_future(self, prices: pd.Series) -> None:
        y = make_direction_target(prices, horizon=5)
        # The last `horizon` bars have no knowable label and must be dropped.
        assert len(y) == len(prices) - 5
        fwd = prices.shift(-5) / prices - 1.0
        assert y.loc[prices.index[100]] == (1 if fwd.iloc[100] > 0 else 0)

    def test_deadband_produces_hold_class(self, prices: pd.Series) -> None:
        y = make_direction_target(prices, horizon=1, deadband=0.02)
        assert set(np.unique(y.dropna())).issubset({0, 1, 2})
        assert y.value_counts().get(2, 0) > 0, "deadband never fired in 1200 bars"

    def test_regression_target_length(self, prices: pd.Series) -> None:
        assert len(make_regression_target(prices, horizon=3)) == len(prices) - 3

    def test_indicators_are_finite(self, ohlcv: pd.DataFrame) -> None:
        ind = technical_indicators(ohlcv)
        for col in ("ret_1", "rsi_14", "atr_14", "macd", "vol_20"):
            assert np.isfinite(ind[col].dropna()).all(), col

    def test_rsi_within_bounds(self, ohlcv: pd.DataFrame) -> None:
        rsi = technical_indicators(ohlcv)["rsi_14"].dropna()
        assert rsi.between(0, 100).all()

    def test_module_contains_no_backfill(self) -> None:
        """`bfill` would inject future prices; the pipeline must not contain it."""
        import inspect

        import mlfin_core

        source = inspect.getsource(mlfin_core)
        assert ".bfill(" not in source
        assert 'method="bfill"' not in source


class TestPreprocessing:
    def test_fitted_statistics_come_only_from_the_fit_split(self, ohlcv: pd.DataFrame) -> None:
        """The stored centre/scale must be a function of the training rows alone."""
        X = make_feature_frame(ohlcv).iloc[:400]
        train, test = X.iloc[:300], X.iloc[300:]
        pre = FoldSafePreprocessor().fit(train)
        np.testing.assert_allclose(pre.center_.to_numpy(), train.median().to_numpy())
        np.testing.assert_allclose(pre.scale_.to_numpy(), train.std(ddof=0).fillna(1.0).to_numpy())
        # Adding the test rows to the *fit* data must move the statistics --
        # this is exactly the leak FoldSafePreprocessor exists to avoid.
        contaminated = FoldSafePreprocessor().fit(pd.concat([train, test]))
        assert not np.allclose(pre.center_.to_numpy(), contaminated.center_.to_numpy())

    def test_transform_is_affine_and_shift_equivariant(self, ohlcv: pd.DataFrame) -> None:
        X = make_feature_frame(ohlcv).iloc[:300]
        train, test = X.iloc[:200], X.iloc[200:]
        pre = FoldSafePreprocessor().fit(train)
        base = pre.transform(test).to_numpy()
        shifted = pre.transform(test + 5.0).to_numpy()
        scale = pre.scale_.to_numpy()
        np.testing.assert_allclose(shifted, base + 5.0 / scale, rtol=1e-8, atol=1e-8)

    def test_leaky_variant_differs_from_safe_variant(self, ohlcv: pd.DataFrame) -> None:
        X = make_feature_frame(ohlcv).iloc[:300]
        train, test = X.iloc[:200], X.iloc[200:]
        pre = FoldSafePreprocessor().fit(train)
        safe_te = pre.transform(test).to_numpy()
        _, leaky_te = FoldSafePreprocessor.fit_on_full_sample(train, test)
        assert not np.allclose(safe_te, leaky_te)

    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            FoldSafePreprocessor().transform(pd.DataFrame({"a": [1.0]}))

    def test_constant_column_does_not_divide_by_zero(self) -> None:
        X = pd.DataFrame({"a": np.ones(50), "b": np.arange(50.0)})
        assert np.isfinite(FoldSafePreprocessor().fit_transform(X).to_numpy()).all()


class TestSplits:
    def test_walk_forward_is_ordered_and_non_overlapping(self) -> None:
        splits = walk_forward_splits(2000, n_splits=5, min_train_size=500, test_size=200)
        assert len(splits) == 5
        prev_test_end = -1
        for train, test in splits:
            assert train.max() < test.min(), "training must precede test"
            assert test.min() > prev_test_end, "test blocks must be disjoint"
            prev_test_end = test.max()

    def test_walk_forward_embargo_gap_exists(self) -> None:
        """Training must stop `embargo` bars before the test block opens."""
        embargo = 10
        splits = walk_forward_splits(2000, n_splits=4, min_train_size=600, test_size=200, embargo=embargo)
        for train, test in splits:
            assert train.max() <= test.min() - embargo - 1

    def test_expanding_grows_and_sliding_is_bounded(self) -> None:
        expanding = walk_forward_splits(3000, n_splits=5, min_train_size=500, test_size=200)
        sliding = walk_forward_splits(
            3000, n_splits=5, min_train_size=500, test_size=200, expanding=False, train_window=400
        )
        sizes = [len(tr) for tr, _ in expanding]
        assert sizes == sorted(sizes)
        assert all(len(tr) <= 400 for tr, _ in sliding)

    def test_walk_forward_rejects_impossible_scheme(self) -> None:
        with pytest.raises(ValueError):
            walk_forward_splits(300, n_splits=20, min_train_size=250, test_size=100)

    def test_purged_kfold_covers_every_observation_as_test(self) -> None:
        splits = purged_kfold_splits(1000, n_splits=5, t1=0, embargo=0)
        covered = np.concatenate([te for _, te in splits])
        assert sorted(covered.tolist()) == list(range(1000))

    def test_purging_removes_overlapping_labels(self) -> None:
        """With horizon t1, no training label may reach into the test block."""
        n_obs, t1 = 600, 20
        splits = purged_kfold_splits(n_obs, n_splits=5, t1=t1, embargo=0)
        for train, test in splits:
            test_mask = np.zeros(n_obs, dtype=bool)
            test_mask[test] = True
            leaks = test_mask[np.clip(train + t1, 0, n_obs - 1)] | test_mask[train]
            assert not leaks.any(), "purging failed to remove label-window overlap"

    def test_embargo_is_enforced(self) -> None:
        n_obs, embargo = 1000, 15
        for train, test in purged_kfold_splits(n_obs, n_splits=5, t1=0, embargo=embargo):
            forbidden = set(range(int(test.max()) + 1, min(int(test.max()) + 1 + embargo, n_obs)))
            assert not forbidden.intersection(train.tolist())

    def test_purging_shrinks_training_set(self) -> None:
        plain = purged_kfold_splits(1000, n_splits=5, t1=0, embargo=0)
        purged = purged_kfold_splits(1000, n_splits=5, t1=30, embargo=0)
        assert sum(len(tr) for tr, _ in purged) < sum(len(tr) for tr, _ in plain)

    def test_purged_kfold_rejects_bad_args(self) -> None:
        with pytest.raises(ValueError):
            purged_kfold_splits(100, n_splits=1)
        with pytest.raises(ValueError):
            purged_kfold_splits(100, n_splits=5, t1=-1)

    def test_cpcv_generates_expected_number_of_paths(self) -> None:
        from math import comb

        paths = combinatorial_purged_cv_splits(1200, n_groups=6, n_test_groups=2, t1=5)
        assert len(paths) == comb(6, 2)
        for path in paths:
            assert len(path) == 2
            starts = [int(te[0]) for _, te in path]
            assert starts == sorted(starts), "path blocks must be chronological"

    def test_cpcv_rejects_degenerate_config(self) -> None:
        with pytest.raises(ValueError):
            combinatorial_purged_cv_splits(600, n_groups=4, n_test_groups=4)


class TestExecutionAndCosts:
    def test_position_is_lagged_one_bar(self, prices: pd.Series) -> None:
        """Invariant #3: a position cannot earn on the bar that produced it."""
        pos = np.ones(len(prices))
        pos[:10] = 0.0
        r = positions_to_returns(pd.Series(pos, index=prices.index), prices, cost_bps=0.0)
        first_held = int(np.argmax(pos == 1.0))
        assert r.iloc[first_held] == pytest.approx(0.0), "took a return on its own signal bar"
        assert r.iloc[first_held + 1] != 0.0

    def test_perfect_foresight_is_reproducible(self, prices: pd.Series) -> None:
        """Accounting sanity check: an oracle beats every realisable model."""
        oracle = np.sign(prices.pct_change().fillna(0.0).to_numpy())
        r = positions_to_returns(oracle, prices, cost_bps=0.0, lag=0)
        assert r.mean() > 0
        assert sharpe_ratio(r) > 0

    def test_costs_reduce_performance(self, prices: pd.Series) -> None:
        pos = pd.Series(np.sign(prices.pct_change().fillna(0.0).to_numpy()), index=prices.index)
        assert positions_to_returns(pos, prices, 50.0).sum() < positions_to_returns(pos, prices, 0.0).sum()

    def test_costs_punish_flipping(self, prices: pd.Series) -> None:
        flip = pd.Series(([1, -1] * (len(prices) // 2))[: len(prices)], index=prices.index, dtype=float)
        hold = pd.Series(1.0, index=prices.index)
        assert positions_to_returns(flip, prices, 100.0).sum() < positions_to_returns(hold, prices, 100.0).sum()

    def test_cash_position_yields_zero(self, prices: pd.Series) -> None:
        r = positions_to_returns(pd.Series(0.0, index=prices.index), prices, cost_bps=25.0)
        assert np.allclose(r.to_numpy(), 0.0)

    def test_buy_and_hold_matches_price_series(self, prices: pd.Series) -> None:
        bh = buy_and_hold_returns(prices)
        np.testing.assert_allclose(bh.iloc[1:].to_numpy(), prices.pct_change().dropna().to_numpy())

    def test_execution_stays_in_valid_position_space(self) -> None:
        probs = np.array([[0.2, 0.7, 0.1], [0.2, 0.7, 0.1], [0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
        pos = apply_causal_execution(probs)
        assert len(pos) == len(probs)
        assert set(np.unique(pos)).issubset({-1.0, 0.0, 1.0})

    def test_reversal_requires_confirmation(self) -> None:
        """A flip needs `confirm_bars` consecutive bars wanting the new side."""
        long = np.array([[0.05, 0.9, 0.05]] * 3)
        short = np.array([[0.9, 0.05, 0.05]] * 3)
        pos = apply_causal_execution(np.vstack([long, short]), confirm_bars=2)
        # Reversal first appears at bar 3, but only commits at bar 4.
        assert pos.tolist() == [1.0, 1.0, 1.0, 1.0, -1.0, -1.0]

    def test_low_confidence_abstains(self) -> None:
        """Chance level for 3 classes is 1/3, so the default 0.40 can abstain."""
        probs = np.array([[0.36, 0.33, 0.31]] * 5)
        assert np.allclose(apply_causal_execution(probs), 0.0)

    def test_confident_prediction_does_trade(self) -> None:
        """Default class map is (sell=0, buy=1, hold=2), so buy mass goes on 1."""
        probs = np.array([[0.15, 0.60, 0.25]] * 3)
        assert np.allclose(apply_causal_execution(probs), 1.0)


class TestMetrics:
    def test_sharpe_of_zero_vol_series_is_undefined(self) -> None:
        """No dispersion means no Sharpe; returning a huge number would be a lie."""
        assert not np.isfinite(sharpe_ratio(pd.Series(np.zeros(200))))

    def test_sharpe_of_constant_returns_is_undefined(self) -> None:
        assert not np.isfinite(sharpe_ratio(pd.Series(np.full(200, 0.001))))

    def test_sortino_of_zero_downside_vol_is_nan(self) -> None:
        assert not np.isfinite(sortino_ratio(pd.Series(np.full(500, 0.002))))

    def test_max_drawdown_on_known_path(self) -> None:
        # equity 1.5 -> 0.75 -> 0.375 -> 0.4125, so the trough drawdown is -75%.
        assert max_drawdown(pd.Series([0.5, -0.5, -0.5, 0.1])) == pytest.approx(-0.75)

    def test_profit_factor(self) -> None:
        assert profit_factor(pd.Series([0.10, 0.10, -0.05])) == pytest.approx(4.0)

    def test_performance_report_is_complete(self) -> None:
        r = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 750))
        rep = performance_report(r)
        for key in ("total_return", "sharpe", "sortino", "max_drawdown", "costs_bps", "n_obs"):
            assert key in rep
        assert rep["n_obs"] == 750
        assert np.isfinite(rep["sharpe"])

    def test_performance_report_is_empty_safe(self) -> None:
        assert performance_report(pd.Series([], dtype=float))["n_obs"] == 0


class TestUncertainty:
    def test_block_bootstrap_shape_and_range(self) -> None:
        idx = block_bootstrap_indices(500, 20, 100, np.random.default_rng(0))
        assert idx.shape == (100, 500)
        assert idx.min() >= 0 and idx.max() < 500

    def test_block_bootstrap_produces_runs(self) -> None:
        """Consecutive draws must come in runs of the block length."""
        idx = block_bootstrap_indices(100, 10, 5, np.random.default_rng(1))
        offsets = np.diff(idx[0, :10])
        assert all(o == 1 for o in offsets if o != 0)

    def test_block_bootstrap_rejects_bad_block(self) -> None:
        with pytest.raises(ValueError):
            block_bootstrap_indices(100, 0, 5, np.random.default_rng(0))

    def test_bootstrap_ci_brackets_point_estimate(self) -> None:
        r = pd.Series(np.random.default_rng(5).normal(0.0008, 0.012, 1500))
        ci = bootstrap_sharpe_ci(r, n_boot=1500, block_size=15, seed=1)
        assert ci["lo"] < ci["point"] < ci["hi"]

    def test_block_ci_is_no_narrower_than_iid(self) -> None:
        """The reason for block bootstrapping: autocorrelation must widen the CI."""
        r = pd.Series(np.random.default_rng(7).normal(0.001, 0.015, 1000))
        block = bootstrap_sharpe_ci(r, n_boot=2000, block_size=30, seed=2)
        iid = bootstrap_sharpe_ci(r, n_boot=2000, block_size=1, seed=2)
        assert (block["hi"] - block["lo"]) >= (iid["hi"] - iid["lo"])

    def test_bootstrap_ci_widens_as_sample_shrinks(self) -> None:
        rng = np.random.default_rng(9)
        full = bootstrap_sharpe_ci(pd.Series(rng.normal(0.001, 0.015, 2000)), 2000, 20, seed=3)
        short = bootstrap_sharpe_ci(pd.Series(rng.normal(0.001, 0.015, 200)), 2000, 20, seed=3)
        assert (short["hi"] - short["lo"]) > (full["hi"] - full["lo"])

    def test_bootstrap_handles_tiny_sample(self) -> None:
        ci = bootstrap_sharpe_ci(pd.Series(np.random.default_rng(0).normal(0, 0.01, 10)))
        assert ci["n_boot"] == 0 and np.isnan(ci["lo"])

    def test_psr_rises_with_more_evidence(self) -> None:
        rng = np.random.default_rng(13)
        small = pd.Series(rng.normal(0.0012, 0.012, 250))
        large = pd.Series(rng.normal(0.0012, 0.012, 3000))
        assert probabilistic_sharpe_ratio(large) > probabilistic_sharpe_ratio(small)

    def test_psr_high_for_strong_stable_edge(self) -> None:
        r = pd.Series(np.random.default_rng(3).normal(0.004, 0.008, 2500))
        assert probabilistic_sharpe_ratio(r) > 0.95

    def test_psr_near_half_for_pure_noise(self) -> None:
        r = pd.Series(np.random.default_rng(4).normal(0.0, 0.01, 2000))
        assert 0.30 < probabilistic_sharpe_ratio(r) < 0.70

    def test_deflated_sharpe_is_penalised_by_many_trials(self) -> None:
        """The whole point: identical P&L looks weaker after a big search.

        The edge is deliberately moderate so that neither PSR nor DSR saturates
        at 1.0 -- saturation would hide exactly the effect under test.
        """
        r = pd.Series(np.random.default_rng(21).normal(0.0006, 0.012, 2000))
        few = deflated_sharpe_ratio(r, n_trials=1)
        many = deflated_sharpe_ratio(r, n_trials=500)
        assert few["sr_benchmark"] == 0.0
        assert many["sr_benchmark"] > 0.0
        assert few["sr"] == pytest.approx(many["sr"])
        assert 0.0 < few["psr"] < 1.0, "test signal too weak to be informative"
        assert many["dsr"] < few["dsr"]

    def test_deflated_sharpe_handles_nan(self) -> None:
        assert deflated_sharpe_ratio(pd.Series([0.1, 0.2]))["dsr"] is not None

    def test_min_track_record_length_is_finite_for_positive_sharpe(self) -> None:
        r = pd.Series(np.random.default_rng(8).normal(0.002, 0.01, 1500))
        n_star = min_track_record_length(r)
        assert np.isfinite(n_star) and n_star > 0

    def test_min_track_record_length_is_nan_without_edge(self) -> None:
        r = pd.Series(np.random.default_rng(2).normal(-0.001, 0.01, 500))
        assert np.isnan(min_track_record_length(r))


class TestCalibration:
    def test_reliability_table_shape(self) -> None:
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, 5000)
        y = (rng.uniform(size=5000) < p).astype(float)
        table = reliability_table(y, p, n_bins=10)
        assert not table.empty
        assert set(["bin", "n", "pred_mean", "obs_freq", "gap"]).issubset(table.columns)
        assert table["n"].sum() == 5000

    def test_ece_is_zero_for_perfectly_calibrated_input(self) -> None:
        rng = np.random.default_rng(1)
        p = rng.uniform(0.05, 0.95, 40000)
        y = (rng.uniform(size=40000) < p).astype(float)
        assert expected_calibration_error(y, p, n_bins=10) < 0.01

    def test_ece_detects_overconfidence(self) -> None:
        rng = np.random.default_rng(2)
        y = (rng.uniform(size=4000) < 0.5).astype(float)
        assert expected_calibration_error(y, np.full(4000, 0.95), n_bins=10) > 0.4

    def test_brier_score_is_minimised_by_the_truth(self) -> None:
        y = np.array([0, 1, 2, 1])
        onehot = np.eye(3)[y]
        assert brier_skill_score(y, onehot) == pytest.approx(1.0)

    def test_brier_skill_score_is_negative_for_bad_forecast(self) -> None:
        y = np.array([0, 1, 2, 1, 0, 2])
        wrong = np.eye(3)[(y + 1) % 3]
        assert brier_skill_score(y, wrong) < 0.0


class TestConformal:
    def test_split_conformal_is_symmetric_around_point_forecast(self) -> None:
        lo, hi = split_conformal_interval(
            np.array([0.01, -0.02, 0.03, 0.00]), np.array([0.0, 0.0, 0.0, 0.0])
        )
        assert lo == -hi

    def test_conformal_interval_widens_with_alpha(self) -> None:
        resid = np.random.default_rng(0).normal(0, 0.01, 800)
        pred = np.zeros_like(resid)
        assert split_conformal_interval(resid, pred, alpha=0.2)[1] < split_conformal_interval(
            resid, pred, alpha=0.01
        )[1]

    def test_conformal_coverage_is_at_least_nominal(self) -> None:
        """Finite-sample guarantee: no distributional assumption is used."""
        rng = np.random.default_rng(5)
        n = 2000
        y = rng.standard_t(df=3, size=n) * 0.01  # fat tails, non-normal
        pred = np.zeros(n)
        lo, hi = split_conformal_interval(y[:1000], pred[:1000], alpha=0.1)
        stats_out = conformal_coverage(y[1000:], pred[1000:], np.full(1000, lo), np.full(1000, hi))
        assert stats_out["empirical_coverage"] >= 0.88
        assert stats_out["n"] == 1000

    def test_conformal_coverage_reported_correctly(self) -> None:
        y = np.array([0.0, 1.0, 2.0, 3.0])
        out = conformal_coverage(y, y, np.zeros(4), np.full(4, 3.0))
        assert out["empirical_coverage"] == pytest.approx(1.0)
        assert out["mean_width"] == pytest.approx(3.0)

    def test_conformal_rejects_empty_calibration(self) -> None:
        with pytest.raises(ValueError):
            split_conformal_interval(np.array([]), np.array([]))
