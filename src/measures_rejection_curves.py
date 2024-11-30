import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

# Source: https://github.com/Shifts-Project/shifts


def calc_uncertainty_rejection_curve(
    errors: np.ndarray, uncertainty: np.ndarray, group_by_uncertainty: bool = True
) -> np.ndarray:
    """
    Calculate the mean error for retention in range [0, 1].

    Args:
        errors (np.ndarray): Per-sample errors (shape: [n_samples]).
        uncertainty (np.ndarray): Uncertainties associated with each prediction (shape: [n_samples]).
        group_by_uncertainty (bool): Whether to group errors by uncertainty.

    Returns:
        np.ndarray: The mean error for retention in range [0, 1].
    """
    n_objects = errors.shape[0]
    if group_by_uncertainty:
        data = pd.DataFrame({"errors": errors, "uncertainty": uncertainty})
        mean_errors = data.groupby("uncertainty")["errors"].mean().rename("mean_errors")
        data = data.merge(mean_errors, on="uncertainty")
        errors = data["mean_errors"].sort_index(ascending=True).to_numpy()
    else:
        uncertainty_order = np.argsort(uncertainty)
        errors = errors[uncertainty_order]

    error_rates = np.zeros(n_objects + 1)
    error_rates[:-1] = np.cumsum(errors[::-1]) / n_objects
    return error_rates


def calc_aucs(errors: np.ndarray, uncertainty: np.ndarray):
    """
    Calculate uncertainty rejection AUCs and rejection ratio.

    Args:
        errors (np.ndarray): Per-sample errors (shape: [n_samples]).
        uncertainty (np.ndarray): Uncertainties associated with each prediction (shape: [n_samples]).

    Returns:
        tuple[float, float]: (rejection_ratio, uncertainty_rejection_auc).
    """
    uncertainty_rejection_curve = calc_uncertainty_rejection_curve(errors, uncertainty)
    uncertainty_rejection_auc = auc(
        np.arange(len(uncertainty_rejection_curve)) / len(uncertainty_rejection_curve),
        uncertainty_rejection_curve,
    )
    random_rejection_auc = uncertainty_rejection_curve[0] / 2
    ideal_rejection_curve = calc_uncertainty_rejection_curve(errors, errors)
    ideal_rejection_auc = auc(
        np.arange(len(ideal_rejection_curve)) / len(ideal_rejection_curve),
        ideal_rejection_curve,
    )

    rejection_ratio = (
        (uncertainty_rejection_auc - random_rejection_auc) / (ideal_rejection_auc - random_rejection_auc) * 100.0
    )
    return rejection_ratio, uncertainty_rejection_auc


def prr_classification(labels: np.ndarray, probs: np.ndarray, measure: np.ndarray, rev: bool):
    """
    Calculate rejection AUCs for classification using uncertainty or other measures.

    Args:
        labels (np.ndarray): Ground truth labels (shape: [n_samples]).
        probs (np.ndarray): Model probabilities (shape: [n_samples, n_classes]).
        measure (np.ndarray): Measure values (shape: [n_samples]).
        rev (bool): Whether to reverse the measure.

    Returns:
        tuple[float, float]: (rejection_ratio, uncertainty_rejection_auc).
    """
    if rev:
        measure = -measure
    preds = np.argmax(probs, axis=1)
    errors = (labels != preds).astype(float)
    return calc_aucs(errors, measure)


def ood_detect(
    domain_labels: np.ndarray,
    in_measure: np.ndarray,
    out_measure: np.ndarray,
    mode: str = "ROC",
    pos_label: int = 1,
):
    """
    Perform out-of-distribution (OOD) detection.

    Args:
        domain_labels (np.ndarray): Ground truth domain labels (shape: [n_samples]).
        in_measure (np.ndarray): In-distribution measures (shape: [n_samples]).
        out_measure (np.ndarray): Out-of-distribution measures (shape: [n_samples]).
        mode (str): Evaluation mode ("ROC" or "PR").
        pos_label (int): Label for positive class (default: 1).

    Returns:
        float: AUC score (ROC-AUC or PR-AUC).
    """
    scores = np.concatenate((in_measure, out_measure))
    scores = scores.astype(np.float128)

    if pos_label != 1:
        scores *= -1.0

    if mode == "PR":
        precision, recall, _ = precision_recall_curve(domain_labels, scores)
        return auc(recall, precision)
    elif mode == "ROC":
        return roc_auc_score(domain_labels, scores)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def f_beta_metrics(errors: np.ndarray, uncertainty: np.ndarray, threshold: float, beta: float = 1.0):
    """
    Calculate F-beta metrics for errors and uncertainties.

    Args:
        errors (np.ndarray): Per-sample errors (shape: [n_samples]).
        uncertainty (np.ndarray): Uncertainties associated with each prediction (shape: [n_samples]).
        threshold (float): Error threshold below which predictions are acceptable.
        beta (float): Beta value for the F-beta metric (default: 1).

    Returns:
        tuple[float, float, np.ndarray]: (F-beta AUC, F-beta at 95%, F-beta scores).
    """
    f_scores, pr, rec = _calc_fbeta_rejection_curve(errors, uncertainty, threshold, beta)
    retention = np.linspace(0, 1, len(pr))
    f_auc = auc(retention[::-1], f_scores)
    f95_index = int(0.95 * len(pr))
    f95 = f_scores[::-1][f95_index]

    return f_auc, f95, f_scores[::-1]


def _acceptable_error(errors: np.ndarray, threshold: float) -> np.ndarray:
    """Return binary array indicating whether errors are below a threshold."""
    return (errors <= threshold).astype(np.float32)


def _calc_fbeta_rejection_curve(
    errors: np.ndarray,
    uncertainty: np.ndarray,
    threshold: float,
    beta: float,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate F-beta scores for a rejection curve.

    Args:
        errors (np.ndarray): Per-sample errors.
        uncertainty (np.ndarray): Prediction uncertainties.
        threshold (float): Error threshold.
        beta (float): Beta value for F-beta calculation.
        eps (float): Small value to prevent division by zero.

    Returns:
        tuple: (F-beta scores, precision, recall).
    """
    acceptable = _acceptable_error(errors, threshold)
    precision, recall, _ = precision_recall_curve(acceptable, -uncertainty)
    f_scores = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + eps)
    return f_scores, precision, recall


# F1 AUC ROC ACC

# func (logits, labels) -> dicts
