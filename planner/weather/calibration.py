"""
Out-of-Sample Probabilistic Calibration & Proper Scoring Rules (`planner/weather/calibration.py`).

Implements leak-free statistical post-processing (EMOS / Ridge + LOO-Conformalized Quantile
Calibration and regularized Logistic Inversion-Breakout Calibration) evaluated strictly
against **independent** physical observations:
  - Transmittance (`P10 / P50 / P90` targeting 80% PI coverage): `NREL NSRDB 2 km 5-min DNI`
  - Coastal summit above marine fog (`P(summit_above_fog)`): NWS `RAOB` radiosondes (`KOAK`, `KVBG`)

Design invariants:
  1. Pure NumPy + stdlib (zero heavy ML dependencies).
  2. Strict out-of-sample cross-validation: supports both **Leave-Site-Out** (spatial holdout)
     and **Chronological Blocked** (temporal holdout) splits — never tunes on test targets.
  3. Residual scale & conformal quantile factors are fitted exclusively on **training-fold
     leave-one-out residuals** (never in-sample residuals), preventing interval under-dispersion.
  4. Never trains or scores against the solver's own `t = 0` internal output (`sunset_ray_truth_pct`).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION_MODEL_PATH = (
    ROOT_DIR / "data" / "weather_benchmarks" / "calibration_model_v1.json"
)


# ---------------------------------------------------------------------------
# 1. Proper Scoring Rules & Calibration Diagnostics
# ---------------------------------------------------------------------------


def pinball_loss(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    quantile: float,
) -> float:
    """Strictly proper pinball (check) loss for quantile `q in (0, 1)`."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.size == 0:
        return 0.0
    err = yt - yp
    return float(np.mean(np.maximum(quantile * err, (quantile - 1.0) * err)))


def crps_from_quantiles(
    y_true: Sequence[float] | np.ndarray,
    p10: Sequence[float] | np.ndarray,
    p50: Sequence[float] | np.ndarray,
    p90: Sequence[float] | np.ndarray,
) -> float:
    """
    Strictly proper quantile-based CRPS approximation (2x average pinball loss across
    `q = 0.10, 0.50, 0.90`, weighted by trapezoidal quantile mass `[0.25, 0.50, 0.25]`).
    Expressed in the same units as `y_true` (percentage points `0..100`). Lower is better.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    if yt.size == 0:
        return 0.0
    l10 = pinball_loss(yt, p10, 0.10)
    l50 = pinball_loss(yt, p50, 0.50)
    l90 = pinball_loss(yt, p90, 0.90)
    return float(2.0 * (0.25 * l10 + 0.50 * l50 + 0.25 * l90))


def interval_coverage_and_width(
    y_true: Sequence[float] | np.ndarray,
    p10: Sequence[float] | np.ndarray,
    p90: Sequence[float] | np.ndarray,
) -> tuple[float, float]:
    """Returns `(empirical_coverage_fraction, mean_interval_width)` for `[p10, p90]`."""
    yt = np.asarray(y_true, dtype=np.float64)
    lo = np.asarray(p10, dtype=np.float64)
    hi = np.asarray(p90, dtype=np.float64)
    if yt.size == 0:
        return 0.0, 0.0
    covered = (yt >= lo - 1e-6) & (yt <= hi + 1e-6)
    return float(np.mean(covered)), float(np.mean(np.maximum(0.0, hi - lo)))


def brier_score(
    y_true_bin: Sequence[int | float] | np.ndarray,
    probs: Sequence[float] | np.ndarray,
) -> float:
    """Strictly proper Brier score for binary events `y in {0, 1}`."""
    yt = np.asarray(y_true_bin, dtype=np.float64)
    pp = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
    if yt.size == 0:
        return 0.0
    return float(np.mean((pp - yt) ** 2))


def brier_skill_score(
    y_true_bin: Sequence[int | float] | np.ndarray,
    probs: Sequence[float] | np.ndarray,
    ref_probs: Sequence[float] | np.ndarray | None = None,
) -> float:
    """
    Brier Skill Score `1 - BS / BS_ref`.
    If `ref_probs` is omitted, uses the empirical base-rate climatology as reference.
    """
    yt = np.asarray(y_true_bin, dtype=np.float64)
    if yt.size == 0:
        return 0.0
    bs = brier_score(yt, probs)
    if ref_probs is None:
        base = float(np.mean(yt))
        bs_ref = float(np.mean((base - yt) ** 2))
    else:
        bs_ref = brier_score(yt, ref_probs)
    if bs_ref <= 1e-9:
        return 0.0
    return float(1.0 - bs / bs_ref)


def expected_calibration_error(
    y_true_bin: Sequence[int | float] | np.ndarray,
    probs: Sequence[float] | np.ndarray,
    n_bins: int = 5,
) -> float:
    """Expected Calibration Error (`ECE`) across `n_bins` probability bins in `[0, 1]`."""
    yt = np.asarray(y_true_bin, dtype=np.float64)
    pp = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
    n = yt.size
    if n == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, max(2, int(n_bins)) + 1)
    ece = 0.0
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        if b == len(edges) - 2:
            mask = (pp >= lo) & (pp <= hi)
        else:
            mask = (pp >= lo) & (pp < hi)
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue
        ece += (cnt / n) * abs(float(np.mean(pp[mask])) - float(np.mean(yt[mask])))
    return float(ece)


def summarize_transmittance_metrics(
    y_true: Sequence[float] | np.ndarray,
    p50: Sequence[float] | np.ndarray,
    p10: Sequence[float] | np.ndarray,
    p90: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Computes full suite of deterministic + probabilistic transmittance verification metrics."""
    yt = np.asarray(y_true, dtype=np.float64)
    ym = np.asarray(p50, dtype=np.float64)
    lo = np.asarray(p10, dtype=np.float64)
    hi = np.asarray(p90, dtype=np.float64)
    if yt.size == 0:
        return {
            "n": 0,
            "mae_pct": 0.0,
            "rmse_pct": 0.0,
            "bias_pct": 0.0,
            "crps_pct": 0.0,
            "pinball_p10": 0.0,
            "pinball_p50": 0.0,
            "pinball_p90": 0.0,
            "pi80_coverage_pct": 0.0,
            "pi80_mean_width_pct": 0.0,
        }
    err = ym - yt
    cov, width = interval_coverage_and_width(yt, lo, hi)
    return {
        "n": int(yt.size),
        "mae_pct": round(float(np.mean(np.abs(err))), 2),
        "rmse_pct": round(float(np.sqrt(np.mean(err ** 2))), 2),
        "bias_pct": round(float(np.mean(err)), 2),
        "crps_pct": round(crps_from_quantiles(yt, lo, ym, hi), 2),
        "pinball_p10": round(pinball_loss(yt, lo, 0.10), 2),
        "pinball_p50": round(pinball_loss(yt, ym, 0.50), 2),
        "pinball_p90": round(pinball_loss(yt, hi, 0.90), 2),
        "pi80_coverage_pct": round(cov * 100.0, 1),
        "pi80_mean_width_pct": round(width, 1),
    }


# ---------------------------------------------------------------------------
# 2. Leak-Free Cross-Validation Splitters
# ---------------------------------------------------------------------------


def leave_group_out_splits(
    groups: Sequence[str],
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """
    Yields `(train_idx, test_idx, group_name)` for Leave-One-Group-Out CV (e.g. site holdout).
    Strictly guarantees zero overlap between training and held-out groups.
    """
    arr = np.asarray(groups, dtype=object)
    unique_groups = sorted(set(str(g) for g in groups))
    splits: list[tuple[np.ndarray, np.ndarray, str]] = []
    for g in unique_groups:
        test_idx = np.nonzero(arr == g)[0]
        train_idx = np.nonzero(arr != g)[0]
        if train_idx.size > 0 and test_idx.size > 0:
            splits.append((train_idx, test_idx, g))
    return splits


def chronological_blocked_splits(
    date_strings: Sequence[str],
    n_folds: int = 4,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """
    Splits observations into `n_folds` contiguous chronological blocks by ISO date so
    temporally adjacent/same-date events never leak across train and test folds.
    """
    unique_dates = sorted(set(str(d) for d in date_strings))
    k = max(2, min(int(n_folds), len(unique_dates)))
    blocks = np.array_split(np.asarray(unique_dates, dtype=object), k)
    date_arr = np.asarray([str(d) for d in date_strings], dtype=object)
    splits: list[tuple[np.ndarray, np.ndarray, str]] = []
    for fold_i, blk in enumerate(blocks):
        blk_set = set(str(x) for x in blk.tolist())
        if not blk_set:
            continue
        test_mask = np.array([d in blk_set for d in date_arr], dtype=bool)
        train_idx = np.nonzero(~test_mask)[0]
        test_idx = np.nonzero(test_mask)[0]
        if train_idx.size > 0 and test_idx.size > 0:
            label = f"block_{fold_i + 1}_{min(blk_set)}_to_{max(blk_set)}"
            splits.append((train_idx, test_idx, label))
    return splits


def cross_year_holdout_splits(
    date_strings: Sequence[str],
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """
    Splits observations by calendar year (Leave-One-Year-Out) for strict cross-year
    temporal generalization (e.g., train on 2024 -> blind test on 2025, and vice versa).
    Returns an empty list when fewer than 2 distinct calendar years are present.
    """
    years = [str(d)[:4] for d in date_strings]
    unique_years = sorted(set(y for y in years if len(y) == 4 and y.isdigit()))
    if len(unique_years) < 2:
        return []
    yr_arr = np.asarray(years, dtype=object)
    splits: list[tuple[np.ndarray, np.ndarray, str]] = []
    for test_yr in unique_years:
        test_idx = np.nonzero(yr_arr == test_yr)[0]
        train_idx = np.nonzero(yr_arr != test_yr)[0]
        if train_idx.size > 0 and test_idx.size > 0:
            train_yrs = "+".join(y for y in unique_years if y != test_yr)
            splits.append((train_idx, test_idx, f"train_{train_yrs}_test_{test_yr}"))
    return splits


# ---------------------------------------------------------------------------
# 3. Transmittance EMOS + Conformalized Quantile Calibrator
# ---------------------------------------------------------------------------

TRANSMITTANCE_FEATURE_NAMES: tuple[str, ...] = (
    "intercept",
    "raw_exp_norm",
    "raw_spread_norm",
    "cloud_low_norm",
    "cloud_mid_norm",
    "cloud_high_norm",
    "is_coastal",
    "log_lead_hr",
)


def build_transmittance_feature_vector(
    raw_exp_pct: float,
    raw_p10_pct: float,
    raw_p90_pct: float,
    cloud_low_pct: float,
    cloud_mid_pct: float,
    cloud_high_pct: float,
    is_coastal: bool,
    lead_hours: float,
) -> np.ndarray:
    """Extracts strictly forecast-time features (zero ground-truth fields)."""
    raw_spread = max(0.0, float(raw_p90_pct) - float(raw_p10_pct))
    return np.array(
        [
            1.0,
            max(0.0, min(1.0, float(raw_exp_pct) / 100.0)),
            max(0.0, min(1.0, raw_spread / 100.0)),
            max(0.0, min(1.0, float(cloud_low_pct) / 100.0)),
            max(0.0, min(1.0, float(cloud_mid_pct) / 100.0)),
            max(0.0, min(1.0, float(cloud_high_pct) / 100.0)),
            1.0 if is_coastal else 0.0,
            math.log1p(max(0.0, float(lead_hours))),
        ],
        dtype=np.float64,
    )


def _bounded_logit(p: np.ndarray, eps: float = 0.015) -> np.ndarray:
    pc = np.clip(p, eps, 1.0 - eps)
    return np.log(pc / (1.0 - pc))


def _bounded_sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -25.0, 25.0)))


@dataclass
class TransmittanceCalibrator:
    """
    Bounded Huber-Weighted Logit + Event-Grouped Conformal Quantile Calibrator for
    golden-hour direct solar transmittance (`0..100%`).

    Physics & Statistical Design:
    1. **Bounded `[0, 100]` logit-space mapping**:
         `p_cal = sigmoid(a0 + a1 * logit(p_raw) + cov_correction)`
       - Respects `[0, 100]` physical transmittance bounds automatically (solid `0%` overcast
         never gets shifted up to `+15%`, and clear `99%` stays near `99%`, while mid-range
         Beer-Lambert slant-path over-attenuation is corrected smoothly).
       - Shrunk toward the physical identity `(a0 = 0, a1 = 1)` via L2 regularization and
         fitted with Huber-weighted IRLS so rare storm-timing outliers do not flatten slope.
       - Automatically enables regularized multi-covariate residual terms (cloud layer mix,
         coastal indicator, lead time) only when `n_unique_events >= 80`, avoiding small-sample
         overfitting on compact archives while scaling cleanly to 500–1,000+ event datasets.
    2. **Event-grouped out-of-fold conformal quantiles**:
       - Computes strictly out-of-event residuals inside the training fold (so multi-lead rows
         from the same physical day never leak into the uncertainty scale), scales by
         heteroscedastic `sigma(p_cal, raw_spread, lead_h)`, and applies finite-sample
         conformal rank multipliers `(1 - alpha/2) * (n_eff + 1) / n_eff`.
    """

    a0: float = 0.0
    a1: float = 1.0
    beta_cov: np.ndarray = field(default_factory=lambda: np.zeros(len(TRANSMITTANCE_FEATURE_NAMES)))
    feat_mean: np.ndarray = field(default_factory=lambda: np.zeros(len(TRANSMITTANCE_FEATURE_NAMES)))
    feat_std: np.ndarray = field(default_factory=lambda: np.ones(len(TRANSMITTANCE_FEATURE_NAMES)))
    use_covariates: bool = False
    l2_a0: float = 1.5
    l2_a1: float = 6.0
    huber_delta: float = 0.25
    sigma_spread_gain: float = 0.35
    sigma_lead_gain: float = 4.0
    sigma_boundary_gain: float = 22.0
    sigma_floor: float = 6.0
    q_low_mult: float = -1.2816
    q_high_mult: float = 1.2816
    target_coverage: float = 0.80
    n_train: int = 0
    n_unique_events: int = 0

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        Xs = X.copy()
        Xs[:, 1:] = (Xs[:, 1:] - self.feat_mean[1:]) / self.feat_std[1:]
        return Xs

    def _scale_vector(
        self, p50_pct: np.ndarray, raw_spread: np.ndarray, lead_hours: np.ndarray
    ) -> np.ndarray:
        p_norm = np.clip(p50_pct / 100.0, 0.0, 1.0)
        boundary_factor = np.sqrt(np.maximum(0.04, p_norm * (1.0 - p_norm)))
        return np.maximum(
            self.sigma_floor,
            self.sigma_spread_gain * np.maximum(0.0, raw_spread)
            + self.sigma_lead_gain * np.log1p(np.maximum(0.0, lead_hours))
            + self.sigma_boundary_gain * boundary_factor,
        )

    @staticmethod
    def _fit_huber_logit_core(
        raw_exp_pct: np.ndarray,
        y_true_pct: np.ndarray,
        event_ids: Sequence[str],
        l2_a0: float,
        l2_a1: float,
        huber_delta: float,
        tier_weights: np.ndarray | None = None,
    ) -> tuple[float, float]:
        z = _bounded_logit(raw_exp_pct / 100.0)
        y = np.clip(y_true_pct / 100.0, 0.0, 1.0)
        X = np.column_stack([np.ones_like(z), z])
        w = np.array([0.0, 1.0], dtype=np.float64)
        w_prior = np.array([0.0, 1.0], dtype=np.float64)
        P = np.diag([float(l2_a0), float(l2_a1)])

        # Event-balanced + tier-reliability sample weights so each unique event counts fairly
        # and Tier 1A 1-min ground pyrheliometers anchor the calibration over nadir satellite DNI.
        uniq_ev = list(dict.fromkeys(event_ids))
        ev_counts: dict[str, int] = {}
        for eid in event_ids:
            ev_counts[eid] = ev_counts.get(eid, 0) + 1
        sw = np.array([1.0 / max(1, ev_counts[eid]) for eid in event_ids], dtype=np.float64)
        if tier_weights is not None and tier_weights.size == sw.size:
            sw = sw * np.maximum(0.05, tier_weights)
        sw *= float(max(1, len(uniq_ev))) / max(1e-9, float(np.sum(sw)))

        for _ in range(30):
            p = _bounded_sigmoid(X @ w)
            err = p - y
            huber_w = np.where(
                np.abs(err) <= huber_delta,
                1.0,
                huber_delta / np.maximum(1e-6, np.abs(err)),
            )
            W = sw * huber_w * np.maximum(1e-4, p * (1.0 - p))
            grad = X.T @ (sw * huber_w * err) + P @ (w - w_prior)
            hess = (X.T * W) @ X + P
            step = np.linalg.solve(hess, grad)
            w -= step
            if float(np.linalg.norm(step)) < 1e-6:
                break

        a0 = float(w[0])
        a1 = float(max(0.45, min(1.85, w[1])))
        return a0, a1

    def _predict_median(self, X: np.ndarray, raw_exp_pct: np.ndarray) -> np.ndarray:
        z = _bounded_logit(raw_exp_pct / 100.0)
        eta = self.a0 + self.a1 * z
        if self.use_covariates and self.beta_cov.size == X.shape[1]:
            Xs = self._standardize(X)
            eta = eta + np.clip(Xs @ self.beta_cov, -1.5, 1.5)
        return 100.0 * _bounded_sigmoid(eta)

    def fit(
        self,
        X: np.ndarray,
        raw_exp_pct: np.ndarray,
        y_true_pct: np.ndarray,
        raw_spread_pct: np.ndarray,
        lead_hours: np.ndarray,
        target_coverage: float = 0.80,
        event_ids: Sequence[str] | None = None,
        tier_weights: Sequence[float] | np.ndarray | None = None,
    ) -> "TransmittanceCalibrator":
        X_arr = np.asarray(X, dtype=np.float64)
        raw_arr = np.asarray(raw_exp_pct, dtype=np.float64)
        y_arr = np.asarray(y_true_pct, dtype=np.float64)
        sp_arr = np.asarray(raw_spread_pct, dtype=np.float64)
        ld_arr = np.asarray(lead_hours, dtype=np.float64)
        tw_arr = (
            np.asarray(tier_weights, dtype=np.float64)
            if tier_weights is not None
            else np.ones(y_arr.size, dtype=np.float64)
        )

        n = int(y_arr.size)
        self.n_train = n
        self.target_coverage = float(target_coverage)
        ev_list = (
            [str(e) for e in event_ids]
            if (event_ids is not None and len(event_ids) == n)
            else [f"row_{i}" for i in range(n)]
        )
        uniq_events = list(dict.fromkeys(ev_list))
        self.n_unique_events = len(uniq_events)

        if n < 3 or self.n_unique_events < 2:
            self.a0 = 0.0
            self.a1 = 1.0
            self.beta_cov = np.zeros(X_arr.shape[1], dtype=np.float64)
            self.use_covariates = False
            return self

        self.feat_mean = np.mean(X_arr, axis=0)
        std = np.std(X_arr, axis=0)
        self.feat_std = np.where(std > 1e-6, std, 1.0)
        self.feat_mean[0] = 0.0
        self.feat_std[0] = 1.0

        # 1. Fit 2-parameter bounded Huber-weighted logit core
        self.a0, self.a1 = self._fit_huber_logit_core(
            raw_arr, y_arr, ev_list, self.l2_a0, self.l2_a1, self.huber_delta, tier_weights=tw_arr
        )

        # 2. Optional covariate logit residual correction only on large archives (N_events >= 80)
        self.beta_cov = np.zeros(X_arr.shape[1], dtype=np.float64)
        self.use_covariates = self.n_unique_events >= 80
        if self.use_covariates:
            Xs = self._standardize(X_arr)
            base_p = _bounded_sigmoid(self.a0 + self.a1 * _bounded_logit(raw_arr / 100.0))
            logit_resid = _bounded_logit(y_arr / 100.0) - _bounded_logit(base_p)
            w_cov = np.maximum(0.05, tw_arr)
            w_cov = w_cov * (float(n) / max(1e-9, float(np.sum(w_cov))))
            reg = np.eye(Xs.shape[1], dtype=np.float64) * 60.0
            reg[0, 0] = 1e6  # intercept already handled by a0
            self.beta_cov = np.linalg.solve((Xs.T * w_cov) @ Xs + reg, Xs.T @ (w_cov * logit_resid))

        # 3. Strictly event-grouped out-of-fold residuals for conformal quantile calibration
        oof_p50 = np.empty(n, dtype=np.float64)
        if self.n_unique_events <= 40:
            # Exact Leave-One-Event-Out inside training fold
            event_folds = [[eid] for eid in uniq_events]
        else:
            # 10-fold event-grouped CV inside training fold
            k_folds = 10
            event_folds = [uniq_events[i::k_folds] for i in range(k_folds)]

        ev_arr = np.array(ev_list, dtype=object)
        for held_events in event_folds:
            val_mask = np.isin(ev_arr, held_events)
            tr_mask = ~val_mask
            if not np.any(tr_mask) or not np.any(val_mask):
                continue
            a0_sub, a1_sub = self._fit_huber_logit_core(
                raw_arr[tr_mask],
                y_arr[tr_mask],
                ev_arr[tr_mask].tolist(),
                self.l2_a0,
                self.l2_a1,
                self.huber_delta,
                tier_weights=tw_arr[tr_mask],
            )
            z_val = _bounded_logit(raw_arr[val_mask] / 100.0)
            oof_p50[val_mask] = 100.0 * _bounded_sigmoid(a0_sub + a1_sub * z_val)

        loo_err = y_arr - oof_p50
        scales = self._scale_vector(oof_p50, sp_arr, ld_arr)
        z_scores = loo_err / scales

        # Finite-sample conformal rank levels using effective number of independent events
        alpha = max(0.02, min(0.45, 1.0 - self.target_coverage))
        alpha_half = 0.5 * alpha
        n_eff = float(max(2, self.n_unique_events))
        q_hi_prob = min(0.99, (1.0 - alpha_half) * (n_eff + 1.0) / n_eff)
        q_lo_prob = max(0.01, 1.0 - q_hi_prob)
        self.q_low_mult = float(np.quantile(z_scores, q_lo_prob))
        self.q_high_mult = float(np.quantile(z_scores, q_hi_prob))
        if self.q_low_mult > -0.15:
            self.q_low_mult = min(self.q_low_mult, -0.35)
        if self.q_high_mult < 0.15:
            self.q_high_mult = max(self.q_high_mult, 0.35)
        return self

    def predict(
        self,
        X: np.ndarray,
        raw_exp_pct: np.ndarray,
        raw_spread_pct: np.ndarray,
        lead_hours: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns calibrated `(p10, p50, p90)` arrays in `[0.0, 100.0]`."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        raw_arr = np.asarray(raw_exp_pct, dtype=np.float64)
        sp_arr = np.asarray(raw_spread_pct, dtype=np.float64)
        ld_arr = np.asarray(lead_hours, dtype=np.float64)

        p50 = np.clip(self._predict_median(X_arr, raw_arr), 0.0, 100.0)
        scales = self._scale_vector(p50, sp_arr, ld_arr)
        p10 = np.clip(p50 + self.q_low_mult * scales, 0.0, p50)
        p90 = np.clip(p50 + self.q_high_mult * scales, p50, 100.0)
        return p10, p50, p90

    def predict_one(
        self,
        raw_exp_pct: float,
        raw_p10_pct: float,
        raw_p90_pct: float,
        cloud_low_pct: float,
        cloud_mid_pct: float,
        cloud_high_pct: float,
        is_coastal: bool,
        lead_hours: float,
    ) -> tuple[float, float, float]:
        feat = build_transmittance_feature_vector(
            raw_exp_pct,
            raw_p10_pct,
            raw_p90_pct,
            cloud_low_pct,
            cloud_mid_pct,
            cloud_high_pct,
            is_coastal,
            lead_hours,
        )
        p10, p50, p90 = self.predict(
            feat,
            np.array([raw_exp_pct], dtype=np.float64),
            np.array([max(0.0, raw_p90_pct - raw_p10_pct)], dtype=np.float64),
            np.array([lead_hours], dtype=np.float64),
        )
        return float(p10[0]), float(p50[0]), float(p90[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "bounded_huber_logit_conformal",
            "feature_names": list(TRANSMITTANCE_FEATURE_NAMES),
            "a0": round(float(self.a0), 6),
            "a1": round(float(self.a1), 6),
            "beta_cov": [round(float(x), 6) for x in self.beta_cov.tolist()],
            "feat_mean": [round(float(x), 6) for x in self.feat_mean.tolist()],
            "feat_std": [round(float(x), 6) for x in self.feat_std.tolist()],
            "use_covariates": bool(self.use_covariates),
            "l2_a0": round(float(self.l2_a0), 3),
            "l2_a1": round(float(self.l2_a1), 3),
            "huber_delta": round(float(self.huber_delta), 3),
            "sigma_spread_gain": round(float(self.sigma_spread_gain), 4),
            "sigma_lead_gain": round(float(self.sigma_lead_gain), 4),
            "sigma_boundary_gain": round(float(self.sigma_boundary_gain), 4),
            "sigma_floor": round(float(self.sigma_floor), 4),
            "q_low_mult": round(float(self.q_low_mult), 4),
            "q_high_mult": round(float(self.q_high_mult), 4),
            "target_coverage": round(float(self.target_coverage), 3),
            "n_train": int(self.n_train),
            "n_unique_events": int(self.n_unique_events),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TransmittanceCalibrator":
        return cls(
            a0=float(d.get("a0", 0.0)),
            a1=float(d.get("a1", 1.0)),
            beta_cov=np.asarray(
                d.get("beta_cov", np.zeros(len(TRANSMITTANCE_FEATURE_NAMES))),
                dtype=np.float64,
            ),
            feat_mean=np.asarray(
                d.get("feat_mean", np.zeros(len(TRANSMITTANCE_FEATURE_NAMES))),
                dtype=np.float64,
            ),
            feat_std=np.asarray(
                d.get("feat_std", np.ones(len(TRANSMITTANCE_FEATURE_NAMES))),
                dtype=np.float64,
            ),
            use_covariates=bool(d.get("use_covariates", False)),
            l2_a0=float(d.get("l2_a0", 1.5)),
            l2_a1=float(d.get("l2_a1", 6.0)),
            huber_delta=float(d.get("huber_delta", 0.25)),
            sigma_spread_gain=float(d.get("sigma_spread_gain", 0.35)),
            sigma_lead_gain=float(d.get("sigma_lead_gain", 4.0)),
            sigma_boundary_gain=float(d.get("sigma_boundary_gain", 22.0)),
            sigma_floor=float(d.get("sigma_floor", 6.0)),
            q_low_mult=float(d.get("q_low_mult", -1.2816)),
            q_high_mult=float(d.get("q_high_mult", 1.2816)),
            target_coverage=float(d.get("target_coverage", 0.80)),
            n_train=int(d.get("n_train", 0)),
            n_unique_events=int(d.get("n_unique_events", 0)),
        )


# ---------------------------------------------------------------------------
# 4. Probabilistic Inversion / Summit-Above-Fog Breakout Calibrator
# ---------------------------------------------------------------------------


def _sigmoid(z: np.ndarray) -> np.ndarray:
    zc = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-zc))


@dataclass
class InversionClearanceCalibrator:
    """
    Regularized logistic probability calibrator for `P(summit_above_fog_lid)`:
        `z = clearance_ft / sigma_lid_ft(lead_h)`
        `P(above_lid) = sigmoid(w0 + w1 * z)`
    where `sigma_lid_ft(lead_h) = base_ft + slope_ft * sqrt(lead_h)` accounts for
    growing marine-layer lid altitude uncertainty with forecast lead time.
    """

    w0: float = 0.0
    w1: float = 1.65
    sigma_base_ft: float = 280.0
    sigma_lead_ft: float = 140.0
    l2_penalty: float = 1.2
    n_train: int = 0

    def _normalized_margin(
        self, clearance_ft: np.ndarray, lead_hours: np.ndarray
    ) -> np.ndarray:
        sigma = self.sigma_base_ft + self.sigma_lead_ft * np.sqrt(
            np.maximum(0.0, lead_hours)
        )
        return np.clip(clearance_ft / np.maximum(120.0, sigma), -6.0, 6.0)

    def fit(
        self,
        clearance_ft: Sequence[float] | np.ndarray,
        lead_hours: Sequence[float] | np.ndarray,
        y_above_lid: Sequence[int | bool] | np.ndarray,
    ) -> "InversionClearanceCalibrator":
        clr = np.asarray(clearance_ft, dtype=np.float64)
        ld = np.asarray(lead_hours, dtype=np.float64)
        y = np.asarray(y_above_lid, dtype=np.float64)
        self.n_train = int(y.size)
        if y.size < 4 or len(set(y.tolist())) < 2:
            # Keep physical default prior if single-class or tiny sample
            return self

        z = self._normalized_margin(clr, ld)
        X = np.column_stack([np.ones_like(z), z])
        w = np.array([0.0, 1.5], dtype=np.float64)
        lam = float(self.l2_penalty)
        # Prior shrink for w1 toward +1.5 (positive physical monotonicity: higher clearance -> higher prob)
        w_prior = np.array([0.0, 1.5], dtype=np.float64)

        for _ in range(25):
            p = _sigmoid(X @ w)
            W = np.maximum(1e-5, p * (1.0 - p))
            grad = X.T @ (p - y) + lam * (w - w_prior)
            hess = (X.T * W) @ X + np.eye(2) * lam
            step = np.linalg.solve(hess, grad)
            w -= step
            if float(np.linalg.norm(step)) < 1e-6:
                break

        self.w0 = float(w[0])
        self.w1 = float(max(0.25, w[1]))
        return self

    def predict_proba(
        self,
        clearance_ft: Sequence[float] | np.ndarray,
        lead_hours: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        clr = np.asarray(clearance_ft, dtype=np.float64)
        ld = np.asarray(lead_hours, dtype=np.float64)
        z = self._normalized_margin(clr, ld)
        return _sigmoid(self.w0 + self.w1 * z)

    def predict_one(self, clearance_ft: float, lead_hours: float) -> float:
        return float(
            self.predict_proba(
                np.array([clearance_ft], dtype=np.float64),
                np.array([lead_hours], dtype=np.float64),
            )[0]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "w0": round(float(self.w0), 5),
            "w1": round(float(self.w1), 5),
            "sigma_base_ft": round(float(self.sigma_base_ft), 1),
            "sigma_lead_ft": round(float(self.sigma_lead_ft), 1),
            "l2_penalty": round(float(self.l2_penalty), 3),
            "n_train": int(self.n_train),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InversionClearanceCalibrator":
        return cls(
            w0=float(d.get("w0", 0.0)),
            w1=float(d.get("w1", 1.65)),
            sigma_base_ft=float(d.get("sigma_base_ft", 280.0)),
            sigma_lead_ft=float(d.get("sigma_lead_ft", 140.0)),
            l2_penalty=float(d.get("l2_penalty", 1.2)),
            n_train=int(d.get("n_train", 0)),
        )


# ---------------------------------------------------------------------------
# 5. Out-of-Sample Benchmark Evaluation & Model Persistence
# ---------------------------------------------------------------------------

MODE_TO_LEAD_HOURS: dict[str, float] = {
    "nowcast_1h": 1.0,
    "nowcast_3h": 3.0,
    "nwp_only": 6.0,
    "hindcast_24h": 24.0,
    "hindcast_72h": 72.0,
}


TIER_1A_SURFRAD_WEIGHT = 1.0
TIER_1B_NSRDB_OPEN_WEIGHT = 0.45
TERRAIN_SHADOW_HORIZON_THRESHOLD_DEG = 3.5


def is_flat_nsrdb_terrain_shadow_artifact(
    truth_tier: str,
    terrain_horizon_deg: float,
    blocker_label: str,
) -> bool:
    """
    Returns True when a flat-horizon 2 km satellite NSRDB ground-truth measurement (`tier_1b_nsrdb_2km`)
    is paired against a steep mountain/canyon tripod where PhotoPlanner's 3D curved-Earth DEM raycaster
    correctly detects topographic ridge occlusion (`terrain_horizon_deg >= 3.5°` or `blocker == terrain`).
    Flat satellite DNI ignores local canyon shadows and reports ~99% clear sky while the tripod is in shadow,
    so excluding these rows prevents calibration from artificially inflating transmittance predictions.
    """
    return (
        truth_tier == "tier_1b_nsrdb_2km"
        and (
            float(terrain_horizon_deg) >= TERRAIN_SHADOW_HORIZON_THRESHOLD_DEG
            or str(blocker_label).strip().lower() == "terrain"
        )
    )


def extract_independent_records_from_benchmark(
    events_detail: Sequence[dict[str, Any]],
    modes: Sequence[str] = ("nowcast_1h", "nowcast_3h", "nwp_only", "hindcast_24h", "hindcast_72h"),
    exclude_flat_nsrdb_terrain_shadows: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Extracts strictly **independent** ground-truth records from `backtest_summary.json`
    `events_detail`:
      1. `trans_records`: verified ONLY against Tier 1A SURFRAD 1-min pyrheliometers or
         open-horizon Tier 1B NSRDB 2 km satellite DNI (excluding flat-horizon NSRDB 3D mountain
         shadow artifacts by default).
      2. `raob_records`: verified ONLY against `raob_ref.summit_truly_above_lid` (NWS radiosonde).
    """
    trans_records: list[dict[str, Any]] = []
    raob_records: list[dict[str, Any]] = []

    for ev in events_detail:
        site = str(ev.get("site") or "unknown")
        region = str(ev.get("region") or "coastal")
        is_coastal = region == "coastal"
        date_str = str(ev.get("date") or "")
        surfrad_ref = ev.get("surfrad_ref")
        if isinstance(surfrad_ref, dict) and surfrad_ref.get("golden_hour_rel_transmittance_pct") is not None:
            nsrdb_y = surfrad_ref.get("golden_hour_rel_transmittance_pct")
            truth_tier = "tier_1a_surfrad"
        else:
            nsrdb_y = ev.get("nsrdb_2km_golden_truth_pct")
            truth_tier = str(ev.get("primary_truth_tier") or "tier_1b_nsrdb_2km")
        raob_ref = ev.get("raob_ref")
        modes_dict = ev.get("modes") or {}

        for m_name in modes:
            md = modes_dict.get(m_name)
            if not isinstance(md, dict):
                continue
            lead_h = float(MODE_TO_LEAD_HOURS.get(m_name, 6.0))

            if nsrdb_y is not None and md.get("golden_exp_pct") is not None:
                hor_deg = float(md.get("terrain_horizon_deg", 0.0) or 0.0)
                blk_lbl = str(md.get("blocker_label") or "clear_corridor")
                shadow_artifact = is_flat_nsrdb_terrain_shadow_artifact(truth_tier, hor_deg, blk_lbl)
                if exclude_flat_nsrdb_terrain_shadows and shadow_artifact:
                    pass
                else:
                    raw_exp = float(md["golden_exp_pct"])
                    raw_p10 = float(md.get("golden_p10_pct", max(0.0, raw_exp - 15.0)))
                    raw_p90 = float(md.get("golden_p90_pct", min(100.0, raw_exp + 15.0)))
                    tier_w = (
                        TIER_1A_SURFRAD_WEIGHT
                        if truth_tier == "tier_1a_surfrad"
                        else TIER_1B_NSRDB_OPEN_WEIGHT
                    )
                    trans_records.append(
                        {
                            "event_id": str(ev.get("id") or ""),
                            "site": site,
                            "region": region,
                            "is_coastal": is_coastal,
                            "date": date_str,
                            "mode": m_name,
                            "lead_hours": lead_h,
                            "truth_tier": truth_tier,
                            "tier_weight": tier_w,
                            "terrain_horizon_deg": hor_deg,
                            "blocker_label": blk_lbl,
                            "is_terrain_shadow_artifact": shadow_artifact,
                            "raw_exp_pct": raw_exp,
                            "raw_p10_pct": raw_p10,
                            "raw_p90_pct": raw_p90,
                            "cloud_low_pct": float(md.get("cloud_low_pct", 0.0)),
                            "cloud_mid_pct": float(md.get("cloud_mid_pct", 0.0)),
                            "cloud_high_pct": float(md.get("cloud_high_pct", 0.0)),
                            "y_nsrdb_pct": float(nsrdb_y),
                        }
                    )

            if (
                is_coastal
                and isinstance(raob_ref, dict)
                and raob_ref.get("summit_truly_above_lid") is not None
                and md.get("summit_clearance_ft") is not None
            ):
                raob_records.append(
                    {
                        "event_id": str(ev.get("id") or ""),
                        "site": site,
                        "date": date_str,
                        "mode": m_name,
                        "lead_hours": lead_h,
                        "summit_clearance_ft": float(md["summit_clearance_ft"]),
                        "y_above_lid": 1 if bool(raob_ref["summit_truly_above_lid"]) else 0,
                    }
                )

    return trans_records, raob_records


def _run_transmittance_cv(
    records: Sequence[dict[str, Any]],
    splits: Sequence[tuple[np.ndarray, np.ndarray, str]],
    target_coverage: float = 0.80,
) -> dict[str, Any]:
    """Runs strict out-of-sample CV for `TransmittanceCalibrator` on the supplied splits."""
    if not records or not splits:
        return {"n_samples": 0}

    X = np.vstack(
        [
            build_transmittance_feature_vector(
                r["raw_exp_pct"],
                r["raw_p10_pct"],
                r["raw_p90_pct"],
                r["cloud_low_pct"],
                r["cloud_mid_pct"],
                r["cloud_high_pct"],
                r["is_coastal"],
                r["lead_hours"],
            )
            for r in records
        ]
    )
    raw_exp = np.array([r["raw_exp_pct"] for r in records], dtype=np.float64)
    raw_p10 = np.array([r["raw_p10_pct"] for r in records], dtype=np.float64)
    raw_p90 = np.array([r["raw_p90_pct"] for r in records], dtype=np.float64)
    raw_spread = np.maximum(0.0, raw_p90 - raw_p10)
    lead_h = np.array([r["lead_hours"] for r in records], dtype=np.float64)
    y_true = np.array([r["y_nsrdb_pct"] for r in records], dtype=np.float64)
    tier_w = np.array([float(r.get("tier_weight", 1.0)) for r in records], dtype=np.float64)
    tiers = [str(r.get("truth_tier") or "tier_1b_nsrdb_2km") for r in records]
    modes = [r["mode"] for r in records]
    event_ids = [r["event_id"] for r in records]

    n = len(records)
    oof_p10 = np.full(n, np.nan, dtype=np.float64)
    oof_p50 = np.full(n, np.nan, dtype=np.float64)
    oof_p90 = np.full(n, np.nan, dtype=np.float64)
    by_fold: dict[str, Any] = {}

    for tr_idx, te_idx, fold_label in splits:
        tr_ev_ids = [event_ids[int(i)] for i in tr_idx]
        cal = TransmittanceCalibrator().fit(
            X[tr_idx],
            raw_exp[tr_idx],
            y_true[tr_idx],
            raw_spread[tr_idx],
            lead_h[tr_idx],
            target_coverage=target_coverage,
            event_ids=tr_ev_ids,
            tier_weights=tier_w[tr_idx],
        )
        p10_t, p50_t, p90_t = cal.predict(
            X[te_idx], raw_exp[te_idx], raw_spread[te_idx], lead_h[te_idx]
        )
        oof_p10[te_idx] = p10_t
        oof_p50[te_idx] = p50_t
        oof_p90[te_idx] = p90_t
        if len(splits) <= 6:
            by_fold[fold_label] = {
                "n_train": int(tr_idx.size),
                "n_test": int(te_idx.size),
                "raw_uncalibrated": summarize_transmittance_metrics(
                    y_true[te_idx], raw_exp[te_idx], raw_p10[te_idx], raw_p90[te_idx]
                ),
                "out_of_sample_calibrated": summarize_transmittance_metrics(
                    y_true[te_idx], p50_t, p10_t, p90_t
                ),
            }

    valid = ~np.isnan(oof_p50)
    raw_overall = summarize_transmittance_metrics(
        y_true[valid], raw_exp[valid], raw_p10[valid], raw_p90[valid]
    )
    cal_overall = summarize_transmittance_metrics(
        y_true[valid], oof_p50[valid], oof_p10[valid], oof_p90[valid]
    )

    by_mode: dict[str, Any] = {}
    for m_name in sorted(set(modes)):
        m_mask = valid & np.array([m == m_name for m in modes], dtype=bool)
        if not np.any(m_mask):
            continue
        by_mode[m_name] = {
            "raw_uncalibrated": summarize_transmittance_metrics(
                y_true[m_mask], raw_exp[m_mask], raw_p10[m_mask], raw_p90[m_mask]
            ),
            "out_of_sample_calibrated": summarize_transmittance_metrics(
                y_true[m_mask], oof_p50[m_mask], oof_p10[m_mask], oof_p90[m_mask]
            ),
        }

    by_truth_tier: dict[str, Any] = {}
    for t_name in sorted(set(tiers)):
        t_mask = valid & np.array([t == t_name for t in tiers], dtype=bool)
        if not np.any(t_mask):
            continue
        by_truth_tier[t_name] = {
            "n": int(np.sum(t_mask)),
            "raw_uncalibrated": summarize_transmittance_metrics(
                y_true[t_mask], raw_exp[t_mask], raw_p10[t_mask], raw_p90[t_mask]
            ),
            "out_of_sample_calibrated": summarize_transmittance_metrics(
                y_true[t_mask], oof_p50[t_mask], oof_p10[t_mask], oof_p90[t_mask]
            ),
        }

    out: dict[str, Any] = {
        "n_samples": int(np.sum(valid)),
        "n_folds": len(splits),
        "raw_uncalibrated": raw_overall,
        "out_of_sample_calibrated": cal_overall,
        "by_mode": by_mode,
        "by_truth_tier": by_truth_tier,
    }
    if by_fold:
        out["by_fold"] = by_fold
    return out


def _run_inversion_cv(
    raob_records: Sequence[dict[str, Any]],
    splits: Sequence[tuple[np.ndarray, np.ndarray, str]],
) -> dict[str, Any]:
    """Runs strict out-of-sample CV for `InversionClearanceCalibrator` against RAOB truth."""
    if not raob_records or not splits:
        return {"n_samples": 0}

    clr = np.array([r["summit_clearance_ft"] for r in raob_records], dtype=np.float64)
    ld = np.array([r["lead_hours"] for r in raob_records], dtype=np.float64)
    y = np.array([r["y_above_lid"] for r in raob_records], dtype=np.float64)

    # Naive uncalibrated deterministic rule (step function: clearance > 0 -> 1.0 else 0.0)
    naive_p = (clr > 0.0).astype(np.float64)
    oof_p = np.full(len(raob_records), np.nan, dtype=np.float64)

    for tr_idx, te_idx, _ in splits:
        cal = InversionClearanceCalibrator().fit(clr[tr_idx], ld[tr_idx], y[tr_idx])
        oof_p[te_idx] = cal.predict_proba(clr[te_idx], ld[te_idx])

    valid = ~np.isnan(oof_p)
    yv = y[valid]
    pv = oof_p[valid]
    nv = naive_p[valid]

    return {
        "n_samples": int(np.sum(valid)),
        "n_folds": len(splits),
        "raw_deterministic_step": {
            "brier_score": round(brier_score(yv, nv), 4),
            "brier_skill_score_vs_climo": round(brier_skill_score(yv, nv), 3),
            "accuracy_pct": round(float(np.mean((nv >= 0.5) == (yv >= 0.5))) * 100.0, 1),
            "ece": round(expected_calibration_error(yv, nv, n_bins=5), 4),
        },
        "out_of_sample_calibrated": {
            "brier_score": round(brier_score(yv, pv), 4),
            "brier_skill_score_vs_climo": round(brier_skill_score(yv, pv), 3),
            "accuracy_pct": round(float(np.mean((pv >= 0.5) == (yv >= 0.5))) * 100.0, 1),
            "ece": round(expected_calibration_error(yv, pv, n_bins=5), 4),
        },
    }


def evaluate_calibration_out_of_sample(
    events_detail: Sequence[dict[str, Any]],
    target_coverage: float = 0.80,
    eval_modes: Sequence[str] | None = None,
    exclude_flat_nsrdb_terrain_shadows: bool = True,
) -> dict[str, Any]:
    """
    Executes full out-of-sample probabilistic calibration verification across:
      - **Leave-Site-Out CV** (spatial generalization to unseen locations)
      - **Chronological Blocked CV** (temporal generalization across seasons/regimes)
      - **Cross-Year Holdout CV** (strict out-of-year validation, e.g. train 2024 -> test 2025)
    strictly against independent SURFRAD 1-min (weight 1.0), open-horizon NSRDB 2 km satellite DNI
    (weight 0.45, excluding steep 3D mountain shadow rows where flat NSRDB ignores ridge occlusion),
    and NWS RAOB radiosondes.
    Also fits the final production calibrators on all available clean independent records.
    """
    modes_use = (
        eval_modes
        if eval_modes is not None
        else ("nowcast_1h", "nowcast_3h", "hindcast_24h", "hindcast_72h", "nwp_only")
    )
    all_trans_records, _ = extract_independent_records_from_benchmark(
        events_detail,
        modes=modes_use,
        exclude_flat_nsrdb_terrain_shadows=False,
    )
    trans_records, raob_records = extract_independent_records_from_benchmark(
        events_detail,
        modes=modes_use,
        exclude_flat_nsrdb_terrain_shadows=exclude_flat_nsrdb_terrain_shadows,
    )
    n_excluded_shadow = max(0, len(all_trans_records) - len(trans_records))

    site_splits_t = leave_group_out_splits([r["site"] for r in trans_records])
    chrono_splits_t = chronological_blocked_splits([r["date"] for r in trans_records], n_folds=4)
    year_splits_t = cross_year_holdout_splits([r["date"] for r in trans_records])

    site_splits_r = leave_group_out_splits([r["site"] for r in raob_records])
    chrono_splits_r = chronological_blocked_splits([r["date"] for r in raob_records], n_folds=4)
    year_splits_r = cross_year_holdout_splits([r["date"] for r in raob_records])

    # Fit final production models on all clean independent records (Tier 1A anchored)
    prod_trans = TransmittanceCalibrator()
    if trans_records:
        X_all = np.vstack(
            [
                build_transmittance_feature_vector(
                    r["raw_exp_pct"],
                    r["raw_p10_pct"],
                    r["raw_p90_pct"],
                    r["cloud_low_pct"],
                    r["cloud_mid_pct"],
                    r["cloud_high_pct"],
                    r["is_coastal"],
                    r["lead_hours"],
                )
                for r in trans_records
            ]
        )
        raw_exp_all = np.array([r["raw_exp_pct"] for r in trans_records], dtype=np.float64)
        raw_sp_all = np.array(
            [max(0.0, r["raw_p90_pct"] - r["raw_p10_pct"]) for r in trans_records],
            dtype=np.float64,
        )
        lead_all = np.array([r["lead_hours"] for r in trans_records], dtype=np.float64)
        y_all = np.array([r["y_nsrdb_pct"] for r in trans_records], dtype=np.float64)
        ev_all = [r["event_id"] for r in trans_records]
        tw_all = np.array([float(r.get("tier_weight", 1.0)) for r in trans_records], dtype=np.float64)
        prod_trans.fit(
            X_all,
            raw_exp_all,
            y_all,
            raw_sp_all,
            lead_all,
            target_coverage=target_coverage,
            event_ids=ev_all,
            tier_weights=tw_all,
        )

    prod_inv = InversionClearanceCalibrator()
    if raob_records:
        prod_inv.fit(
            [r["summit_clearance_ft"] for r in raob_records],
            [r["lead_hours"] for r in raob_records],
            [r["y_above_lid"] for r in raob_records],
        )

    return {
        "schema_version": "cal_v2_tier_weighted",
        "independent_ground_truth_only": True,
        "exclude_flat_nsrdb_terrain_shadows": bool(exclude_flat_nsrdb_terrain_shadows),
        "n_excluded_flat_nsrdb_terrain_shadow_rows": int(n_excluded_shadow),
        "tier_weights": {
            "tier_1a_surfrad": TIER_1A_SURFRAD_WEIGHT,
            "tier_1b_nsrdb_open": TIER_1B_NSRDB_OPEN_WEIGHT,
        },
        "transmittance_target": (
            "Tier 1A NOAA SURFRAD 1-min ground pyrheliometers (w=1.0) + "
            "open-horizon Tier 1B NREL NSRDB 2km GOES DNI (w=0.45, steep 3D shadow rows excluded)"
        ),
        "inversion_target": "NWS RAOB Radiosonde (summit_truly_above_lid)",
        "target_pi_coverage_pct": round(target_coverage * 100.0, 1),
        "transmittance_leave_site_out_cv": _run_transmittance_cv(
            trans_records, site_splits_t, target_coverage=target_coverage
        ),
        "transmittance_chronological_cv": _run_transmittance_cv(
            trans_records, chrono_splits_t, target_coverage=target_coverage
        ),
        "transmittance_cross_year_cv": _run_transmittance_cv(
            trans_records, year_splits_t, target_coverage=target_coverage
        ),
        "inversion_leave_site_out_cv": _run_inversion_cv(raob_records, site_splits_r),
        "inversion_chronological_cv": _run_inversion_cv(raob_records, chrono_splits_r),
        "inversion_cross_year_cv": _run_inversion_cv(raob_records, year_splits_r),
        "fitted_models": {
            "transmittance": prod_trans.to_dict(),
            "inversion_clearance": prod_inv.to_dict(),
        },
    }


def save_calibration_bundle(
    report: dict[str, Any],
    path: Path = DEFAULT_CALIBRATION_MODEL_PATH,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def load_calibration_bundle(
    path: Path = DEFAULT_CALIBRATION_MODEL_PATH,
) -> tuple[TransmittanceCalibrator, InversionClearanceCalibrator] | None:
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        fm = doc.get("fitted_models") or {}
        t_cal = TransmittanceCalibrator.from_dict(fm["transmittance"])
        i_cal = InversionClearanceCalibrator.from_dict(fm["inversion_clearance"])
        return t_cal, i_cal
    except Exception:
        return None
