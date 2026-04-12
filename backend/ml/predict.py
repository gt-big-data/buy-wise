"""
backend/ml/predict.py

Inference wrapper for trained LightGBM price forecast models.

Responsibilities:
    - Load trained 7d and 14d models + scaler params (once, at startup)
    - Accept a single row of features (real-time) or a DataFrame (batch)
    - Run inference and inverse-transform predictions back to dollar values
    - Track latency per prediction (mean, median, p95, p99)
    - Return output compatible with classifier.py (predicted_price_7d, predicted_price_14d)

Usage:
    from backend.ml.predict import PricePredictor

    predictor = PricePredictor()   # loads models once

    # Single prediction (real-time — batch_size=1)
    result = predictor.predict_single(feature_row, asin="B00001R3W3")
    print(result.predicted_price_7d)   # dollars
    print(result.predicted_price_14d)  # dollars
    print(result.latency_ms)

    # Batch prediction
    results = predictor.predict_batch(feature_df)

    # Feed directly into classifier
    from backend.ml.classifier import PriceRecommendationClassifier
    clf = PriceRecommendationClassifier()
    recommendation = clf.classify(
        predicted_price_7d  = result.predicted_price_7d,
        predicted_price_14d = result.predicted_price_14d,
        predicted_price_30d = result.predicted_price_14d,  # 14d proxy for 30d
        price_history       = price_history,
    )
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from backend.ml.features import FEATURE_COLS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODEL_DIR = Path(__file__).resolve().parent / "models"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """
    Output for a single prediction.

    predicted_price_7d / 14d are in real dollar values (inverse-transformed).
    Pass these directly into PriceRecommendationClassifier.classify().

    Note on predicted_price_30d:
        No 30d model is trained. predicted_price_14d is used as a proxy.
        This is documented here and in classifier calls so it can be replaced
        if a 30d model is added later.
    """
    asin:                str
    predicted_price_7d:  float          # dollars
    predicted_price_14d: float          # dollars
    predicted_price_30d: float          # dollars — 14d proxy, see note above
    normalized_pred_7d:  float          # raw model output (normalized space)
    normalized_pred_14d: float          # raw model output (normalized space)
    latency_ms:          float          # end-to-end inference time in ms
    inverse_transform_applied: bool     # False if ASIN missing from scaler_params


@dataclass
class BatchPredictionResult:
    """Output for batch inference."""
    predictions:    list[PredictionResult]
    latency_stats:  dict = field(default_factory=dict)   # mean, median, p95, p99

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "asin":                 r.asin,
                "predicted_price_7d":  r.predicted_price_7d,
                "predicted_price_14d": r.predicted_price_14d,
                "predicted_price_30d": r.predicted_price_30d,
                "latency_ms":          r.latency_ms,
            }
            for r in self.predictions
        ])


# ---------------------------------------------------------------------------
# Latency tracker
# ---------------------------------------------------------------------------

class LatencyTracker:
    """
    Tracks inference latency across predictions.
    Used by evaluate.py to compute mean, median, p95, p99.
    """

    def __init__(self):
        self._latencies: list[float] = []

    def record(self, latency_ms: float):
        self._latencies.append(latency_ms)

    def stats(self) -> dict:
        if not self._latencies:
            return {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "n": 0}

        arr = np.array(self._latencies)
        return {
            "mean":   round(float(np.mean(arr)),              3),
            "median": round(float(np.median(arr)),            3),
            "p95":    round(float(np.percentile(arr, 95)),    3),
            "p99":    round(float(np.percentile(arr, 99)),    3),
            "n":      len(self._latencies),
        }

    def reset(self):
        self._latencies = []

    @property
    def meets_latency_target(self) -> bool:
        """True if p99 latency is under 200ms (acceptance criteria gate)."""
        s = self.stats()
        return s["p99"] < 200.0


# ---------------------------------------------------------------------------
# Main predictor
# ---------------------------------------------------------------------------

class PricePredictor:
    """
    Loads trained LightGBM models once and provides inference methods.

    Design:
        - Models are loaded at __init__ time so repeated calls don't reload from disk.
        - Inverse transform converts normalized predictions → dollar values per ASIN.
        - LatencyTracker accumulates timings across calls for evaluate.py.

    Inverse transform formula (matches normalization applied during feature engineering):
        price_dollars = (normalized_prediction * global_norm_sd) + global_mean
    """

    def __init__(
        self,
        model_dir: Path = MODEL_DIR,
        verbose:   bool = True,
    ):
        self.model_dir      = Path(model_dir)
        self.verbose        = verbose
        self.latency_tracker = LatencyTracker()

        self._model_7d:      Optional[lgb.Booster] = None
        self._model_14d:     Optional[lgb.Booster] = None
        self._scaler_params: dict = {}

        self._load_models()

    # -----------------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------------

    def _load_models(self):
        """Load both models and scaler params from disk."""
        path_7d      = self.model_dir / "lgbm_7d.lgb"
        path_14d     = self.model_dir / "lgbm_14d.lgb"
        scaler_path  = self.model_dir / "scaler_params.json"

        if not path_7d.exists():
            raise FileNotFoundError(
                f"7d model not found at {path_7d}\n"
                f"Run: python -m backend.ml.train --horizon 7"
            )
        if not path_14d.exists():
            raise FileNotFoundError(
                f"14d model not found at {path_14d}\n"
                f"Run: python -m backend.ml.train --horizon 14"
            )
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"scaler_params.json not found at {scaler_path}\n"
                f"Run: python -m backend.ml.train  (generates scaler params from train set)"
            )

        t0 = time.perf_counter()

        self._model_7d  = lgb.Booster(model_file=str(path_7d))
        self._model_14d = lgb.Booster(model_file=str(path_14d))

        with open(scaler_path) as f:
            self._scaler_params = json.load(f)

        load_ms = (time.perf_counter() - t0) * 1000

        if self.verbose:
            print(f"[PricePredictor] Models loaded in {load_ms:.1f}ms")
            print(f"[PricePredictor] Scaler params: {len(self._scaler_params)} ASINs")

    # -----------------------------------------------------------------------
    # Inverse transform
    # -----------------------------------------------------------------------

    def _inverse_transform(
        self,
        asin:            str,
        normalized_pred: float,
    ) -> tuple[float, bool]:
        """
        Convert normalized model output back to dollar value.

        Returns:
            (price_dollars, transform_applied)
            transform_applied is False if ASIN not in scaler_params —
            in that case normalized_pred is returned as-is with a warning.
        """
        params = self._scaler_params.get(asin)

        if params is None:
            if self.verbose:
                print(
                    f"[PricePredictor] Warning: ASIN '{asin}' not in scaler_params. "
                    f"Returning normalized value — dollar amounts will be incorrect."
                )
            return round(float(normalized_pred), 4), False

        global_mean     = float(params["global_mean"])
        global_norm_sd  = float(params["global_norm_sd"])

        # Guard: if sd is 0 (constant-price product), prediction = mean
        if global_norm_sd == 0:
            return round(global_mean, 2), True

        price_dollars = (normalized_pred * global_norm_sd) + global_mean

        # Clip to non-negative — models can occasionally predict slightly negative
        # normalized values for very low-priced items
        price_dollars = max(price_dollars, 0.01)

        return round(price_dollars, 2), True

    # -----------------------------------------------------------------------
    # Feature alignment
    # -----------------------------------------------------------------------

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure feature matrix matches the exact columns and order expected by the model.
        Missing columns are filled with 0 (safe default for normalized features).
        Extra columns are dropped.
        """
        available = [c for c in FEATURE_COLS if c in X.columns]
        missing   = [c for c in FEATURE_COLS if c not in X.columns]

        if missing and self.verbose:
            print(f"[PricePredictor] Warning: {len(missing)} feature(s) missing, filling with 0: {missing}")

        # Add missing columns as 0
        for col in missing:
            X = X.copy()
            X[col] = 0.0

        X = X[FEATURE_COLS].astype(float)  # enforce exact column order
        return X

    # -----------------------------------------------------------------------
    # Single prediction (primary real-time path)
    # -----------------------------------------------------------------------

    def predict_single(
        self,
        feature_row: dict | pd.Series | pd.DataFrame,
        asin:        str,
    ) -> PredictionResult:
        """
        Predict 7d and 14d prices for a single product row.

        Args:
            feature_row: Feature values as dict, Series, or single-row DataFrame.
                         Must contain the columns defined in features.FEATURE_COLS.
            asin:        Product ASIN — used for inverse transform lookup.

        Returns:
            PredictionResult with dollar-value predictions and latency_ms.

        Latency target: < 200ms (p99 across calls)
        """
        t_start = time.perf_counter()

        # Normalise input to single-row DataFrame
        if isinstance(feature_row, dict):
            X = pd.DataFrame([feature_row])
        elif isinstance(feature_row, pd.Series):
            X = feature_row.to_frame().T.reset_index(drop=True)
        else:
            if len(feature_row) != 1:
                raise ValueError(
                    f"predict_single expects exactly 1 row, got {len(feature_row)}. "
                    f"Use predict_batch for multiple rows."
                )
            X = feature_row.reset_index(drop=True)

        X_aligned = self._align_features(X)

        # Inference
        norm_pred_7d  = float(self._model_7d.predict(X_aligned)[0])
        norm_pred_14d = float(self._model_14d.predict(X_aligned)[0])

        # Inverse transform to dollars
        price_7d,  transform_applied_7d  = self._inverse_transform(asin, norm_pred_7d)
        price_14d, transform_applied_14d = self._inverse_transform(asin, norm_pred_14d)

        # 14d is proxy for 30d (no 30d model trained)
        price_30d = price_14d

        latency_ms = (time.perf_counter() - t_start) * 1000
        self.latency_tracker.record(latency_ms)

        return PredictionResult(
            asin                     = asin,
            predicted_price_7d       = price_7d,
            predicted_price_14d      = price_14d,
            predicted_price_30d      = price_30d,
            normalized_pred_7d       = round(norm_pred_7d,  4),
            normalized_pred_14d      = round(norm_pred_14d, 4),
            latency_ms               = round(latency_ms,    3),
            inverse_transform_applied = transform_applied_7d and transform_applied_14d,
        )

    # -----------------------------------------------------------------------
    # Batch prediction (evaluate.py / backend batch processing)
    # -----------------------------------------------------------------------

    def predict_batch(
        self,
        df:   pd.DataFrame,
        asin_col: str = "asin",
    ) -> BatchPredictionResult:
        """
        Predict prices for all rows in a DataFrame.

        Args:
            df:       DataFrame with feature columns + 'asin' column.
            asin_col: Name of the ASIN column (default: 'asin').

        Returns:
            BatchPredictionResult with per-row PredictionResult list
            and aggregate latency stats.

        Note:
            Internally this runs row-by-row to match real-time latency
            characteristics (batch_size=1 is the primary deployment scenario).
            For bulk offline evaluation, a vectorised path is provided below.
        """
        if asin_col not in df.columns:
            raise ValueError(
                f"Column '{asin_col}' not found in DataFrame. "
                f"Required for per-ASIN inverse transform."
            )

        self.latency_tracker.reset()
        predictions = []

        for _, row in df.iterrows():
            asin   = str(row[asin_col])
            result = self.predict_single(row, asin=asin)
            predictions.append(result)

        stats = self.latency_tracker.stats()

        if self.verbose:
            print(
                f"[PricePredictor] Batch complete: {len(predictions)} predictions  |  "
                f"mean={stats['mean']}ms  median={stats['median']}ms  "
                f"p95={stats['p95']}ms  p99={stats['p99']}ms"
            )
            if not self.latency_tracker.meets_latency_target:
                print(
                    f"[PricePredictor] ⚠ p99 latency {stats['p99']}ms exceeds 200ms target"
                )

        return BatchPredictionResult(
            predictions   = predictions,
            latency_stats = stats,
        )

    def predict_batch_vectorized(
        self,
        df:       pd.DataFrame,
        asin_col: str = "asin",
    ) -> BatchPredictionResult:
        """
        Vectorized batch inference — faster for large offline evaluation sets.
        Runs both models over the entire DataFrame in two calls instead of N*2.
        Latency stats reflect total batch time divided by N (average per-row).

        Use this in evaluate.py for throughput benchmarking.
        Use predict_batch for realistic per-request latency benchmarking.
        """
        if asin_col not in df.columns:
            raise ValueError(f"Column '{asin_col}' not found in DataFrame.")

        X_aligned = self._align_features(df.copy())
        asins     = df[asin_col].astype(str).values

        t_start = time.perf_counter()

        norm_preds_7d  = self._model_7d.predict(X_aligned)
        norm_preds_14d = self._model_14d.predict(X_aligned)

        total_ms    = (time.perf_counter() - t_start) * 1000
        per_row_ms  = total_ms / max(len(df), 1)

        predictions = []
        for i, asin in enumerate(asins):
            price_7d,  t7  = self._inverse_transform(asin, float(norm_preds_7d[i]))
            price_14d, t14 = self._inverse_transform(asin, float(norm_preds_14d[i]))

            predictions.append(PredictionResult(
                asin                      = asin,
                predicted_price_7d        = price_7d,
                predicted_price_14d       = price_14d,
                predicted_price_30d       = price_14d,
                normalized_pred_7d        = round(float(norm_preds_7d[i]),  4),
                normalized_pred_14d       = round(float(norm_preds_14d[i]), 4),
                latency_ms                = round(per_row_ms, 3),
                inverse_transform_applied = t7 and t14,
            ))

        stats = {
            "mean":          round(per_row_ms, 3),
            "median":        round(per_row_ms, 3),
            "p95":           round(per_row_ms, 3),
            "p99":           round(per_row_ms, 3),
            "n":             len(predictions),
            "total_batch_ms": round(total_ms, 3),
            "note":          "vectorized — per-row stats are averages, not individual measurements",
        }

        if self.verbose:
            print(
                f"[PricePredictor] Vectorized batch: {len(predictions)} rows  |  "
                f"total={total_ms:.1f}ms  avg_per_row={per_row_ms:.2f}ms"
            )

        return BatchPredictionResult(predictions=predictions, latency_stats=stats)

    # -----------------------------------------------------------------------
    # Convenience properties
    # -----------------------------------------------------------------------

    @property
    def latency_stats(self) -> dict:
        """Current accumulated latency stats across all predict_single calls."""
        return self.latency_tracker.stats()

    @property
    def meets_latency_target(self) -> bool:
        """True if p99 latency < 200ms across all recorded predictions."""
        return self.latency_tracker.meets_latency_target

    def reset_latency(self):
        """Clear accumulated latency measurements."""
        self.latency_tracker.reset()


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("Running PricePredictor smoke test...")
    print("(Requires trained models in backend/ml/models/)\n")

    try:
        predictor = PricePredictor()
    except FileNotFoundError as e:
        print(f"Cannot run smoke test: {e}")
        sys.exit(1)

    # Build a dummy feature row with all zeros (just tests the pipeline)
    dummy_row = {col: 0.0 for col in FEATURE_COLS}

    result = predictor.predict_single(dummy_row, asin="B00001R3W3")

    print(f"\nSmoke test result:")
    print(f"  ASIN:                {result.asin}")
    print(f"  predicted_price_7d:  ${result.predicted_price_7d}")
    print(f"  predicted_price_14d: ${result.predicted_price_14d}")
    print(f"  predicted_price_30d: ${result.predicted_price_30d}  (14d proxy)")
    print(f"  latency_ms:          {result.latency_ms}ms")
    print(f"  inverse_transform:   {result.inverse_transform_applied}")

    stats = predictor.latency_stats
    print(f"\nLatency stats: {stats}")
    print(f"Meets <200ms target: {predictor.meets_latency_target}")
