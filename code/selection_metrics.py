"""Decision losses for a fixed candidate set under one common condition."""

from __future__ import annotations

import numpy as np


def selection_metrics(observed, predicted):
    y = np.asarray(observed, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or len(y) < 2:
        raise ValueError("Expected matching vectors with at least two candidates")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Nonfinite response or prediction")
    # This tolerance handles numeric ties only; it is not practical equivalence.
    chosen = np.isclose(p, p.max(), rtol=0.0, atol=1e-12)
    loss = float(y.max() - y[chosen].mean())
    random_loss = float(y.max() - y.mean())
    span = float(np.ptp(y))
    error = p - y
    i, j = np.triu_indices(len(y), k=1)
    observed_difference = y[i] - y[j]
    predicted_difference = p[i] - p[j]
    informative = observed_difference != 0
    prediction_ties = np.abs(predicted_difference) <= 1e-12
    pair_score = np.where(
        prediction_ties, 0.5,
        np.sign(observed_difference) == np.sign(predicted_difference),
    )
    return {
        "candidate_count": len(y), "selected_tie_count": int(chosen.sum()),
        "selection_loss": loss, "random_selection_loss": random_loss,
        "gain_over_random": random_loss - loss,
        "normalized_loss": loss / span if span > 0 else np.nan,
        "observed_range": span,
        "pairwise_accuracy": float(pair_score[informative].mean()) if informative.any() else np.nan,
        "informative_pairs": int(informative.sum()),
        "mae": float(np.abs(error).mean()), "mse": float(np.mean(error ** 2)),
        "common_bias_squared": float(error.mean() ** 2),
        "relative_error_mse": float(np.mean((error - error.mean()) ** 2)),
        "observed_best_selected_probability": float(np.mean(y[chosen] == y.max())),
    }
