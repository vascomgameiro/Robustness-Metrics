import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.utils import assert_all_finite, check_consistent_length, column_or_1d
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.multiclass import type_of_target

# Source: https://github.com/Shifts-Project/shifts


def calc_uncertainty_rejection_curve(errors, uncertainty, group_by_uncertainty=True):
    """
    Calculates the mean error for retention in range [0,1]
    :param errors: Per sample errors - array [n_samples]
    :param uncertainty: Uncertainties associated with each prediction. array [n_samples]
    :param group_by_uncertainty: Whether to group errors by uncertainty
    :return: The mean error for retention in range [0,1]
    """
    n_objects = errors.shape[0]
    if group_by_uncertainty:
        data = pd.DataFrame(dict(errors=errors, uncertainty=uncertainty))
        mean_errors = data.groupby("uncertainty").mean()
        mean_errors.rename(columns={"errors": "mean_errors"}, inplace=True)
        data = data.join(mean_errors, "uncertainty")
        data.drop("errors", axis=1, inplace=True)

        uncertainty_order = data["uncertainty"].argsort()
        errors = data["mean_errors"][uncertainty_order]
    else:
        uncertainty_order = uncertainty.argsort()
        errors = errors[uncertainty_order]

    error_rates = np.zeros(n_objects + 1)
    error_rates[:-1] = np.cumsum(errors)[::-1] / n_objects
    return error_rates


def calc_aucs(errors, uncertainty):
    uncertainty_rejection_curve = calc_uncertainty_rejection_curve(errors, uncertainty)
    uncertainty_rejection_auc = uncertainty_rejection_curve.mean()
    random_rejection_auc = uncertainty_rejection_curve[0] / 2
    ideal_rejection_auc = calc_uncertainty_rejection_curve(errors, errors).mean()

    rejection_ratio = (
        (uncertainty_rejection_auc - random_rejection_auc) / (ideal_rejection_auc - random_rejection_auc) * 100.0
    )
    return rejection_ratio, uncertainty_rejection_auc


def prr_classification(labels, probs, measure, rev: bool):
    if rev:
        measure = -measure
    preds = np.argmax(probs, axis=1)
    errors = (labels != preds).astype("float32")
    return calc_aucs(errors, measure)


def ood_detect(domain_labels, in_measure, out_measure, mode="ROC", pos_label=1):
    scores = np.concatenate((in_measure, out_measure), axis=0)
    scores = np.asarray(scores, dtype=np.float128)
    if pos_label != 1:
        scores *= -1.0

    if mode == "PR":
        precision, recall, thresholds = precision_recall_curve(domain_labels, scores)
        aupr = auc(recall, precision)
        return aupr

    elif mode == "ROC":
        roc_auc = roc_auc_score(domain_labels, scores)
        return roc_auc


def nll_class(target, probs, epsilon=1e-10):
    log_p = -np.log(probs + epsilon)
    return target * log_p[:, 1] + (1 - target) * log_p[:, 0]


def f_beta_metrics(errors, uncertainty, threshold, beta=1.0):
    """
    :param errors: Per sample errors - array [n_samples]
    :param uncertainty: Uncertainties associated with each prediction. array [n_samples]
    :param threshold: The error threshold below which we consider the prediction acceptable
    :param beta: The beta value for the F_beta metric. Defaults to 1
    :return: fbeta_auc, fbeta_95, retention
    """
    f_scores, pr, rec = _calc_fbeta_rejection_curve(errors, uncertainty, threshold, beta)
    ret = np.arange(pr.shape[0]) / pr.shape[0]

    f_auc = auc(ret[::-1], f_scores)
    f95 = f_scores[::-1][np.int(0.95 * pr.shape[0])]

    return f_auc, f95, f_scores[::-1]


def _precision_recall_curve_retention(y_true, probas_pred, *, pos_label=None, sample_weight=None):
    fps, tps, thresholds = _binary_clf_curve_ret(y_true, probas_pred, pos_label=pos_label, sample_weight=sample_weight)

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]

    sl = slice(-1, None, -1)
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def _calc_fbeta_rejection_curve(errors, uncertainty, threshold, beta=1.0, group_by_uncertainty=True, eps=1e-10):
    ae = _acceptable_error(errors, threshold)
    pr, rec, _ = _precision_recall_curve_retention(ae, -uncertainty)
    pr = np.asarray(pr)
    rec = np.asarray(rec)
    f_scores = (1 + beta**2) * pr * rec / (pr * beta**2 + rec + eps)

    return f_scores, pr, rec


def _binary_clf_curve_ret(y_true, y_score, pos_label=None, sample_weight=None):
    y_type = type_of_target(y_true)
    if not (y_type == "binary" or (y_type == "multiclass" and pos_label is not None)):
        raise ValueError("{0} format is not supported".format(y_type))

    check_consistent_length(y_true, y_score, sample_weight)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    assert_all_finite(y_true)
    assert_all_finite(y_score)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)

    pos_label = _check_pos_label_consistency(pos_label, y_true)

    y_true = y_true == pos_label

    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    tps = stable_cumsum(y_true)
    fps = stable_cumsum((1 - y_true))
    return fps, tps, y_score


def get_model_errors(y_pred, y_true, squared=True):
    """
    Get the prediction errors, defined as the differences (or squared differences) of actual target values and the
    model predictions
    :param y_pred: Predictions
    :param y_true: Actual target values
    :param squared: Whether to return the squared difference of y_true and ensemble average predicted output
    :return:
    """
    if not squared:
        errors = y_true - y_pred
    else:
        errors = (y_true - y_pred) ** 2

    return errors


def _check_pos_label_consistency(pos_label, y_true):
    classes = np.unique(y_true)
    if pos_label is None and (
        classes.dtype.kind in "OUS"
        or not (
            np.array_equal(classes, [0, 1])
            or np.array_equal(classes, [-1, 1])
            or np.array_equal(classes, [0])
            or np.array_equal(classes, [-1])
            or np.array_equal(classes, [1])
        )
    ):
        classes_repr = ", ".join(repr(c) for c in classes)
        raise ValueError(
            f"y_true takes value in {{{classes_repr}}} and pos_label is not "
            f"specified: either make y_true take value in {{0, 1}} or "
            f"{{-1, 1}} or pass pos_label explicitly."
        )
    elif pos_label is None:
        pos_label = 1.0

    return pos_label


def _acceptable_error(errors, threshold):
    return np.asarray(errors <= threshold, dtype=np.float32)


def misclassification_error(y_true, y_pred):
    """
    Calculate the misclassification error.

    Parameters:
    - y_true (array-like): Ground truth labels (size: n_samples).
    - y_pred (array-like): Predicted labels (size: n_samples).

    Returns:
    - float: Misclassification error (proportion of incorrect predictions).
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    return (y_true != y_pred).astype(int)
