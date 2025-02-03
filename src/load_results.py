import os
import torch
import pandas as pd


def load_norm_measures(path_to_measures):
    """Load norm measures from a given path."""
    return torch.load(os.path.join(path_to_measures, "norm_measures.pt"))


def extract_measures(norm_measures, keys):
    """Extract specific measures from the norm measures dictionary."""
    return [norm_measures[key] for key in keys]


def create_norm_metrics_dataframe(models_to_train):
    """Create a dataframe with metrics for all models."""
    metrics = {
        "model_name": [],
        "margin": [],
        "inv_margin": [],
        "l1_inf_norm": [],
        "frobenius_norm": [],
        "spectral_norm": [],
        "trace_norm": [],
        "l1_inf_norm_margin": [],
        "frobenius_norm_margin": [],
        "spectral_norm_margin": [],
        "trace_norm_margin": [],
        "l1_inf_norm_squared_margin": [],
        "frobenius_norm_squared_margin": [],
        "spectral_norm_squared_margin": [],
        "trace_norm_squared_margin": [],
        "mu_fro_spec": [],
        "mu_spec_init_main": [],
        "mu_spec_origin_main": [],
        "mu_sum_of_fro": [],
        "mu_sum_of_fro_margin": [],
        "log_product_spectral_norms": [],
        "log_product_frobenius_norms": [],
        "frobenius_distance": [],
        "spectral_distance": [],
        "mu_path_norm": [],
        "mu_path_norm_margin": [],
        "l1_max_bound": [],
        "frobenius_bound": [],
        "spec_l2_1_bound": [],
        "spec_fro_bound": [],
        "flatness": [],
    }

    keys = [
        "Margin",
        "1 / Margin",
        "L_{1,inf} norm",
        "Frobenius norm",
        "Spectral norm",
        "Trace norm",
        "L_{1,inf} norm over margin",
        "Frobenius norm over margin",
        "Spectral norm over margin",
        "Trace norm over margin",
        "L_{1,inf} norm over squared margin",
        "Frobenius norm over squared margin",
        "Spectral norm over squared margin",
        "Trace norm over squared margin",
        "Mu_fro-spec",
        "Mu_spec-init-main",
        "Mu_spec-origin-main",
        "Mu_sum-of-fro",
        "Mu_sum-of-fro/margin",
        "Log Product of Spectral Norms",
        "Log Product of Frobenius Norms",
        "Frobenius Distance",
        "Spectral Distance",
        "Mu path-norm",
        "Mu path-norm/margin",
        "L1_max Bound",
        "Frobenius Bound",
        "Spec_L2_1 Bound",
        "Spec_Fro Bound",
        "Flatness",
    ]

    for config in models_to_train:
        model_name = config["name"]
        path_to_measures = os.path.join(f"models/{model_name}", "measures")
        norm_measures = load_norm_measures(path_to_measures)

        # Append model name
        metrics["model_name"].append(model_name)

        # Extract measures
        measures = extract_measures(norm_measures, keys)

        for key, value in zip(list(metrics.keys())[1:], measures):
            metrics[key].append(value)

    return pd.DataFrame(metrics)


def load_sharpness_measures(path_to_measures):
    """Load sharpness measures from a given path."""
    return torch.load(os.path.join(path_to_measures, "sharpness.pt"), map_location=torch.device("cpu"))


def extract_sharpness_measures(sharpness_measures, keys):
    """Extract specific sharpness measures from the dictionary."""
    return [
        sharpness_measures[key].cpu().item()
        if isinstance(sharpness_measures[key], torch.Tensor)
        else sharpness_measures[key]
        for key in keys
    ]


def create_sharpness_metrics_dataframe(models_to_train):
    """Create a dataframe with sharpness metrics for all models."""
    sharpness_metrics = {
        "model_name": [],
        "PAC_Bayes_Sigma": [],
        "PAC_Bayes_Bound": [],
        "PAC_Bayes_Flatness": [],
        "PAC_Bayes_MAG_Sigma": [],
        "PAC_Bayes_MAG_Bound": [],
        "PAC_Bayes_MAG_Flatness": [],
        "Sharpness_Sigma": [],
        "Sharpness_Bound": [],
        "Sharpness_Flatness": [],
        "Sharpness_MAG_Sigma": [],
        "Sharpness_MAG_Bound": [],
        "Sharpness_MAG_Flatness": [],
    }

    keys = [
        "PAC_Bayes_Sigma",
        "PAC_Bayes_Bound",
        "PAC_Bayes_Flatness",
        "PAC_Bayes_MAG_Sigma",
        "PAC_Bayes_MAG_Bound",
        "PAC_Bayes_MAG_Flatness",
        "Sharpness_Sigma",
        "Sharpness_Bound",
        "Sharpness_Flatness",
        "Sharpness_MAG_Sigma",
        "Sharpness_MAG_Bound",
        "Sharpness_MAG_Flatness",
    ]

    for config in models_to_train:
        model_name = config["name"]
        path_to_measures = os.path.join(f"models/{model_name}", "measures")
        sharpness_measures = load_sharpness_measures(path_to_measures)

        # Append model name
        sharpness_metrics["model_name"].append(model_name)

        # Extract sharpness measures
        measures = extract_sharpness_measures(sharpness_measures, keys)

        for key, value in zip(list(sharpness_metrics.keys())[1:], measures):
            sharpness_metrics[key].append(value)

    return pd.DataFrame(sharpness_metrics)


def load_complexity_measures(path_to_measures):
    """Load complexity measures from a given path."""
    return torch.load(os.path.join(path_to_measures, "complexity_cifar.pt"))


def extract_complexity_measures(complexity_measures, keys):
    """Extract specific performance metrics from the dictionary of interest."""
    selected_dict = complexity_measures.get("complexity_test", {})
    return [selected_dict.get(key, None) for key in keys]


def create_performance_metrics_dataframe(models_to_train):
    """Create a dataframe with performance metrics for all models."""
    performance_metrics = {
        "model_name": [],
        "accuracy": [],
        "top5_accuracy": [],
        "log_loss": [],
        "cross_entropy_loss": [],
        "uncertainty_rejection_auc": [],
        "f1_score": [],
        "recall": [],
        "c_var_risk": [],
        "roc_auc": [],
        "entropy": [],
        "precision": [],
    }

    keys = [
        "Accuracy",
        "Top-5 Accuracy",
        "Log Loss",
        "Cross Entropy Loss",
        "Uncertainty Rejection AUC",
        "F-1 Score",
        "Recall",
        "C-var Risk",
        "ROC AUC",
        "Entropy",
        "Precision",
    ]

    for config in models_to_train:
        model_name = config["name"]
        path_to_measures = os.path.join(f"models/{model_name}", "measures")
        complexity_measures = load_complexity_measures(path_to_measures)

        # Append model name
        performance_metrics["model_name"].append(model_name)

        # Extract performance measures from the specified dictionary
        measures = extract_complexity_measures(complexity_measures, keys)

        for key, value in zip(list(performance_metrics.keys())[1:], measures):
            performance_metrics[key].append(value)

    return pd.DataFrame(performance_metrics)


def extract_ood_complexity_measures(complexity_measures, dataset_name, keys):
    """Extract specific OOD performance metrics from the dictionary of interest."""
    selected_dict = complexity_measures.get(f"complexity_{dataset_name}", {})
    return [selected_dict.get(key, None) for key in keys]


def create_ood_performance_metrics_dataframe(models_to_train, ood_datasets):
    """Create a dataframe with OOD performance metrics for all models and datasets."""
    ood_performance_metrics = {"model_name": []}

    # Dynamically add columns for each OOD dataset and metric
    for dataset in ood_datasets:
        for metric in [
            "accuracy",
            "top5_accuracy",
            "log_loss",
            "cross_entropy_loss",
            "uncertainty_rejection_auc",
            "f1_score",
            "recall",
            "cvar_risk",
            "roc_auc",
            "entropy",
            "precision",
        ]:
            ood_performance_metrics[f"{dataset}_{metric}"] = []

    keys = [
        "Accuracy",
        "Top-5 Accuracy",
        "Log Loss",
        "Cross Entropy Loss",
        "Uncertainty Rejection AUC",
        "F-1 Score",
        "Recall",
        "C-var Risk",
        "ROC AUC",
        "Entropy",
        "Precision",
    ]

    for config in models_to_train:
        model_name = config["name"]
        path_to_measures = os.path.join(f"models/{model_name}", "measures")
        complexity_measures = load_complexity_measures(path_to_measures)

        # Append model name
        ood_performance_metrics["model_name"].append(model_name)

        # Extract OOD performance measures for each dataset
        for dataset in ood_datasets:
            measures = extract_ood_complexity_measures(complexity_measures, dataset, keys)
            for key, value in zip(keys, measures):
                column_name = f"{dataset}_{key.lower().replace(' ', '_').replace('-', '')}"
                if column_name in ood_performance_metrics:
                    ood_performance_metrics[column_name].append(value)
                else:
                    print(key)
                    print(column_name)
                    ood_performance_metrics[column_name] = [None] * len(
                        ood_performance_metrics["model_name"]
                    )
                    ood_performance_metrics[column_name].append(value)
    return pd.DataFrame(ood_performance_metrics)


def load_complexity_differences(path_to_measures):
    """Load complexity differences from a given path."""
    return torch.load(os.path.join(path_to_measures, "complexity_diff.pt"))


def extract_performance_gaps(complexity_differences, dataset_name, keys, gap_type):
    """Extract specific performance gaps (absolute or proportional) from the dictionary."""
    selected_dict = complexity_differences.get(f"complexity_diff_{dataset_name}", {})
    return [selected_dict[metric].get(gap_type, None) for metric in keys]


def create_performance_gap_dataframes(models_to_train, datasets):
    """
    Create dataframes for performance gaps (absolute and proportional).
    """
    # Initialize data dictionaries
    performance_gap_metrics = {"model_name": []}
    proportional_gap_metrics = {"model_name": []}

    # Define the keys (metrics) to extract
    keys = [
        "Accuracy",
        "Top-5 Accuracy",
        "Log Loss",
        "Cross Entropy Loss",
        "Uncertainty Rejection AUC",
        "F-1 Score",
        "Recall",
        "C-var Risk",
        "ROC AUC",
        "Entropy",
        "Precision",
    ]

    # Dynamically add columns for each dataset and metric
    for dataset in datasets:
        for metric in [
            "accuracy",
            "top5_accuracy",
            "log_loss",
            "cross_entropy_loss",
            "uncertainty_rejection_auc",
            "f1_score",
            "recall",
            "cvar_risk",
            "roc_auc",
            "entropy",
            "precision",
        ]:
            performance_gap_metrics[f"{dataset}_{metric}_gap"] = []
            proportional_gap_metrics[f"{dataset}_{metric}_proportional_gap"] = []

    # Iterate through models and extract data
    for config in models_to_train:
        model_name = config["name"]
        path_to_measures = os.path.join(f"models/{model_name}", "measures")
        complexity_differences = load_complexity_differences(path_to_measures)

        # Append model name
        performance_gap_metrics["model_name"].append(model_name)
        proportional_gap_metrics["model_name"].append(model_name)

        # Extract performance gaps for each dataset
        for dataset in datasets:
            # Absolute differences
            absolute_measures = extract_performance_gaps(
                complexity_differences, dataset, keys, "absolute_difference"
            )
            # Proportional differences
            proportional_measures = extract_performance_gaps(
                complexity_differences, dataset, keys, "proportional_difference"
            )

            # Add to the respective dictionaries
            for key, abs_value, prop_value in zip(keys, absolute_measures, proportional_measures):
                abs_column = f"{dataset}_{key.lower().replace(' ', '_').replace('-', '')}_gap"
                prop_column = f"{dataset}_{key.lower().replace(' ', '_').replace('-', '')}_proportional_gap"

                performance_gap_metrics[abs_column].append(abs_value)
                proportional_gap_metrics[prop_column].append(prop_value)

    # Create DataFrames
    performance_gap_df = pd.DataFrame(performance_gap_metrics)
    proportional_performance_gap_df = pd.DataFrame(proportional_gap_metrics)

    return performance_gap_df, proportional_performance_gap_df


def create_train_val_gap_dataframe(models_to_train):
    """
    Create a single dataframe with gaps (absolute, proportional, and elasticity) from train_val.
    """
    # Initialize the data dictionary
    train_val_gap_metrics = {"model_name": []}

    # Define metrics and their suffixes for the columns
    keys = [
        "Accuracy",
        "Top-5 Accuracy",
        "Log Loss",
        "Cross Entropy Loss",
        "Uncertainty Rejection AUC",
        "F-1 Score",
        "Recall",
        "C-var Risk",
        "ROC AUC",
        "Entropy",
        "Precision",
    ]
    suffixes = ["absolute_difference", "proportional_difference", "elasticity"]

    # Dynamically add columns for train_val metrics and gaps
    for metric in [
        "accuracy",
        "top5_accuracy",
        "log_loss",
        "cross_entropy_loss",
        "uncertainty_rejection_auc",
        "f1_score",
        "recall",
        "cvar_risk",
        "roc_auc",
        "entropy",
        "precision",
    ]:
        for suffix in suffixes:
            train_val_gap_metrics[f"train_val_{metric}_{suffix}"] = []

    # Iterate through models to extract data
    for config in models_to_train:
        model_name = config["name"]
        path_to_measures = os.path.join(f"models/{model_name}", "measures")
        complexity_differences = load_complexity_differences(path_to_measures)

        # Append model name
        train_val_gap_metrics["model_name"].append(model_name)

        # Extract train_val gaps
        train_val_differences = complexity_differences.get("complexity_diff_train_val", {})
        for key in keys:
            metric_data = train_val_differences.get(key, {})
            for suffix in suffixes:
                column_name = f"train_val_{key.lower().replace(' ', '_').replace('-', '')}_{suffix}"
                value = metric_data.get(suffix, None)
                if column_name in train_val_gap_metrics:
                    train_val_gap_metrics[column_name].append(value)
                else:
                    train_val_gap_metrics[column_name] = [value]

    train_val_gap_df = pd.DataFrame(train_val_gap_metrics)
    return train_val_gap_df
