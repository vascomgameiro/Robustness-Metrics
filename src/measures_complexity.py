import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, accuracy_score, log_loss

# Sources:
# https://github.com/Shifts-Project/shifts
# https://github.com/facebookresearch/decodable_information_bottleneck
# Fantastic Generalization Measures and Where to Find Them: arXiv:1912.02178


def calculate_top_n_accuracy(y_true, y_pred, n=5, normalize=True):
    """
    Calculate Top-N Accuracy for multiclass classification tasks.
    """
    num_samples = y_pred.shape[0]
    top_n_predictions = np.argsort(y_pred, axis=1)[:, -n:]
    matches = np.array([y_true[i] in top_n_predictions[i] for i in range(num_samples)])
    return matches.mean() if normalize else matches.sum()


def calculate_uncertainty_rejection_curve(errors, uncertainty, group_by_uncertainty=True):
    """
    Calculate the uncertainty rejection curve.
    x-axis: rejection rate
    y-axis: error rate (1-accuracy)
    """
    n_samples = errors.shape[0]
    df = pd.DataFrame({"errors": errors, "uncertainty": uncertainty, "nrsamples": np.ones(n_samples)})

    if group_by_uncertainty:
        df = df.groupby("uncertainty").agg({"errors": "sum", "nrsamples": "sum"})

    df = df.sort_values(by= "uncertainty")
    sample_sizes = df["nrsamples"].to_numpy()
    sample_sizes = np.cumsum(sample_sizes)
    nr_points = len(sample_sizes)
    rejection_rate = np.ones(nr_points+1)
    rejection_rate[1:] = np.ones(nr_points) - sample_sizes / n_samples # in descending order

    errors = df["errors"].to_numpy()
    error_rates = np.zeros(nr_points+1)
    error_rates[1:] = np.cumsum(errors) / sample_sizes #also in descending order

    return error_rates , rejection_rate


def calculate_aucs(errors, uncertainty):
    """
    Calculate AUCs for uncertainty rejection.
    """
    rejection_curve, rejection_rates = calculate_uncertainty_rejection_curve(errors, uncertainty)
    uncertainty_auc = auc(rejection_rates, rejection_curve)
    #random_auc = rejection_curve[-1] / 2
    #ideal_curve = calculate_uncertainty_rejection_curve(errors, errors)
    #ideal_auc = auc(np.linspace(0, 1, len(ideal_curve)), ideal_curve)
    #rejection_ratio = (uncertainty_auc - random_auc) / (ideal_auc - random_auc) * 100
    # -> rejection ratio is not evaluationg the model's performance but the uncertainty measure's performance.... 
    return uncertainty_auc #, rejection_ratio


def calculate_f_beta_metrics(errors, uncertainty, threshold, beta=1.0):
    """
    Calculate F-beta metrics for errors and uncertainties.
    """
    acceptable = (errors <= threshold).astype(float)
    precision, recall, _ = precision_recall_curve(acceptable, -uncertainty)
    f_scores = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-10)
    auc_score = auc(np.linspace(0, 1, len(f_scores)), f_scores) #does not make much sense?
    f95 = f_scores[int(0.95 * len(f_scores))] 
    return auc_score, f95


def perform_ood_detection(domain_labels, in_measure, out_measure, mode="ROC"):
    """
    Perform out-of-distribution detection.
    """
    scores = np.concatenate([in_measure, out_measure]).astype(np.float128)
    labels = np.concatenate([domain_labels, 1 - domain_labels])

    if mode == "PR":
        precision, recall, _ = precision_recall_curve(labels, scores)
        return auc(recall, precision)
    elif mode == "ROC":
        return roc_auc_score(labels, scores)
    else:
        raise ValueError("Unsupported mode: must be 'ROC' or 'PR'")


def calculate_entropy(probs):
    """
    Calculate entropy for probabilistic predictions.
    """
    return -np.sum(probs * np.log(probs + 1e-10), axis=1)


def evaluate_model_metrics(logits, labels, threshold=0.5, beta=1.0):
    """
    Calculate a comprehensive set of metrics for model evaluation, using entropy as the uncertainty measure.
    """
    # Convert logits to probabilities
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    # Predictions and errors
    predictions = np.argmax(probabilities, axis=1)
    errors = (labels != predictions).astype(float)

    # Calculate entropy as uncertainty measure
    entropy = calculate_entropy(probabilities)

    # Metrics calculation
    accuracy = accuracy_score(labels, predictions)
    top_5_accuracy = calculate_top_n_accuracy(labels, probabilities, n=5)
    log_loss_score = log_loss(labels, probabilities)
    uncertainty_auc = calculate_aucs(errors, entropy)
    f_beta_auc, f95 = calculate_f_beta_metrics(errors, entropy, threshold, beta)
    roc_auc = roc_auc_score(labels, probabilities, multi_class="ovr")

    # Cross-entropy loss
    cross_entropy_loss = -np.mean(np.log(probabilities[np.arange(len(labels)), labels] + 1e-10))

    # Compile metrics into a dictionary
    return {
        "Accuracy": accuracy,
        "Top-5 Accuracy": top_5_accuracy,
        "Log Loss": log_loss_score,
        "Cross Entropy Loss": cross_entropy_loss,
        #"Rejection Ratio": rejection_ratio,
        "Uncertainty Rejection AUC": uncertainty_auc,
        "F-Beta AUC": f_beta_auc, #does not make much sense
        "F-Beta at 95%": f95, #don't know what this means, don't know if it is well calculated
        "ROC AUC": roc_auc,
        "Entropy": entropy.mean(),
    }
