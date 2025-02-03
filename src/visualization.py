import matplotlib as mpl
import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd


def set_academic_style(context="notebook", style="whitegrid", palette="deep", font_scale=1.5, rc=None):
    """
    Set the visualization style for Seaborn and Matplotlib to academic standards.

    Parameters:
    - context (str): Scaling of plot elements. Options: paper, notebook, talk, poster.
    - style (str): Seaborn style theme. Options: white, dark, whitegrid, darkgrid, ticks.
    - palette (str or list): Color palette for Seaborn.
    - font_scale (float): Scaling factor for fonts.
    - rc (dict, optional): Additional Matplotlib rc parameters.

    Example:
        set_academic_style()
    """
    # Set Seaborn theme
    sns.set_theme(context=context, style=style, palette=palette, font_scale=font_scale, rc=rc)

    # Define default rc parameters for Matplotlib
    default_rc = {
        "figure.figsize": (6, 4),  # Width, Height in inches
        "axes.labelsize": 14,  # Axis labels
        "axes.titlesize": 16,  # Title size
        "xtick.labelsize": 12,  # X-axis tick labels
        "ytick.labelsize": 12,  # Y-axis tick labels
        "legend.fontsize": 12,  # Legend font size
        "legend.title_fontsize": 14,  # Legend title font size
        "lines.linewidth": 2,  # Line width
        "lines.markersize": 6,  # Marker size
        "font.family": "serif",  # Font family
        "font.serif": ["Times New Roman"],  # Serif font
        "text.usetex": False,  # Use LaTeX for text rendering
        "savefig.dpi": 300,  # Resolution for saved figures
        "savefig.bbox": "tight",  # Bounding box for saved figures
    }

    # Update rcParams with default_rc
    mpl.rcParams.update(default_rc)

    # If additional rc parameters are provided, update them
    if rc:
        mpl.rcParams.update(rc)

    print("Academic visualization style has been set.")


def visualize_test_set_histograms(
    iid_df,
    ood_df,
    metric_column_prefix,
    title_suffix="Metrics",
    iid_label="IID Test Set",
    test_set_filter=None,
    save_path=None,
):
    """
    Visualize histograms for a specified metric across IID and OOD test sets in a grid layout.

    Parameters:
    - iid_df: DataFrame containing IID test set metrics (wide format).
    - ood_df: DataFrame containing OOD test set metrics (wide format).
    - metric_column_prefix: String prefix for the metric columns (e.g., "accuracy", "accuracy_absolute").
    - title_suffix: String to append to the plot titles (default: "Metrics").
    - iid_label: Label for IID test sets (default: "IID Test Set").
    - test_set_filter: List of test set names to filter (default: None, uses all available test sets).
    """
    # Filter columns in both IID and OOD DataFrames based on the prefix, excluding top-5 accuracy
    iid_columns = [
        col
        for col in iid_df.columns
        if metric_column_prefix.lower() in col.lower()
        and "top5" not in col.lower()
        and "model_name" not in col.lower()
    ]
    ood_columns = [
        col
        for col in ood_df.columns
        if metric_column_prefix.lower() in col.lower()
        and "top5" not in col.lower()
        and "model_name" not in col.lower()
    ]

    # Reshape IID and OOD data
    iid_data = iid_df[["model_name"] + iid_columns].melt(
        id_vars="model_name", var_name="Test Set", value_name="Metric"
    )
    ood_data = ood_df[["model_name"] + ood_columns].melt(
        id_vars="model_name", var_name="Test Set", value_name="Metric"
    )

    # Clean column names for better visualization
    iid_data["Test Set"] = (
        iid_data["Test Set"].str.replace(metric_column_prefix, iid_label).str.replace("_", " ").str.title()
    )
    ood_data["Test Set"] = (
        ood_data["Test Set"].str.replace(metric_column_prefix, "").str.replace("_", " ").str.title()
    )

    # Combine IID and OOD data, adding a source column for color coding
    iid_data["Source"] = "IID"
    ood_data["Source"] = "OOD"
    combined_data = pd.concat([iid_data, ood_data])

    # Filter specific test sets if a filter is provided
    if test_set_filter:
        combined_data = combined_data[combined_data["Test Set"].isin(test_set_filter)]

    # Determine grid size
    n_test_sets = len(combined_data["Test Set"].unique())
    n_rows = (n_test_sets // 4) + (n_test_sets % 4 > 0)  # 3x4 layout

    # Plot histograms
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, n_rows * 4))
    axes = axes.flatten()
    for i, test_set in enumerate(combined_data["Test Set"].unique()):
        subset = combined_data[combined_data["Test Set"] == test_set]
        sns.histplot(
            data=subset,
            x="Metric",
            bins=10,
            kde=False,  # Disable KDE to avoid singular matrix error
            ax=axes[i],
            hue="Source",
            palette={"IID": "steelblue", "OOD": "#F08080"},  # Color coding for IID and OOD
        )
        axes[i].set_title(f"{test_set} {title_suffix}")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")
        axes[i].legend(title="Source")
    for j in range(i + 1, len(axes)):  # Turn off unused subplots
        axes[j].axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()


def visualize_test_set_scatter(
    iid_df,
    ood_df,
    iid_metric="accuracy",
    ood_metric_prefix="accuracy",
    title_suffix="Scatter Plots of Test Set Accuracies",
    iid_label="IID Accuracy",
    ood_label="OOD Accuracy",
    test_set_filter=None,
    baseline_line=False,
    suffixes=("vgg", "conv"),
    save_path=None,
):
    """
    Visualize scatter plots of IID accuracy against OOD accuracies for multiple test sets in a grid layout.

    Parameters:
    - iid_df: DataFrame containing IID test set metrics (wide format).
    - ood_df: DataFrame containing OOD test set metrics (wide format).
    - iid_metric: Column name in the IID DataFrame for accuracy (x-axis).
    - ood_metric_prefix: Prefix for metric columns in the OOD DataFrame (y-axis).
    - title_suffix: Title for the scatter plot grid.
    - iid_label: Label for the x-axis (default: "IID Accuracy").
    - ood_label: Label for the y-axis (default: "OOD Accuracy").
    - test_set_filter: List of OOD test sets to filter (default: None, uses all OOD test sets).
    - baseline_line: If True, plot a y = x baseline line on each subplot.
    - suffixes: A tuple/list of suffixes to look for in model names.
                Models that end with one of these suffixes will be colored accordingly.
                Default is ("VGG", "conv").
    - save_path: If specified, saves the figure as a PDF to the given path.
    """

    # Verify the IID metric exists
    if iid_metric not in iid_df.columns:
        raise ValueError(f"'{iid_metric}' not found in IID dataframe columns: {iid_df.columns.tolist()}")

    # Filter OOD metric columns based on the prefix
    ood_columns = [
        col
        for col in ood_df.columns
        if ood_metric_prefix.lower() in col.lower()
        and "top5" not in col.lower()
        and "model_name" not in col.lower()
    ]

    if not ood_columns:
        raise ValueError(f"No columns with prefix '{ood_metric_prefix}' found in OOD dataframe.")

    # Reshape OOD data
    ood_data = ood_df[["model_name"] + ood_columns].melt(
        id_vars="model_name", var_name="Test Set", value_name="OOD Accuracy"
    )

    # Clean column names for better visualization
    ood_data["Test Set"] = (
        ood_data["Test Set"]
        .str.replace(ood_metric_prefix, "", case=False)
        .str.replace("_", " ")
        .str.title()
    )

    # Filter specific OOD test sets if a filter is provided
    if test_set_filter:
        ood_data = ood_data[ood_data["Test Set"].isin(test_set_filter)]

    # Merge IID and OOD data on model_name
    iid_data = iid_df[["model_name", iid_metric]].rename(columns={iid_metric: "IID Accuracy"})
    combined_data = pd.merge(iid_data, ood_data, on="model_name")

    # Create a column that identifies the suffix type
    # e.g., if model_name ends with "VGG", label it "VGG"; if ends with "conv", label it "conv"
    def identify_suffix(model_name):
        # Lowercase comparisons for safety
        lower_name = model_name.lower()
        for sfx in suffixes:
            if lower_name.startswith(sfx.lower()):
                return sfx
        return "other"  # If no suffix matches

    combined_data["Suffix Type"] = combined_data["model_name"].apply(identify_suffix)

    # Determine the axis range (global min/max across all data)
    x_min = combined_data["IID Accuracy"].min() * 0.8
    x_max = combined_data["IID Accuracy"].max() * 1.2
    y_min = combined_data["OOD Accuracy"].min() * 0.8
    y_max = combined_data["OOD Accuracy"].max() * 1.2

    # Determine grid size
    n_test_sets = len(combined_data["Test Set"].unique())
    n_rows = (n_test_sets // 4) + (n_test_sets % 4 > 0)  # number of rows for subplots

    # Create scatter plots in a grid
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, n_rows * 4))
    axes = axes.flatten()

    unique_test_sets = combined_data["Test Set"].unique()

    for i, test_set in enumerate(unique_test_sets):
        subset = combined_data[combined_data["Test Set"] == test_set]
        sns.scatterplot(
            data=subset,
            x="IID Accuracy",
            y="OOD Accuracy",
            hue="Suffix Type",  # <--- color by suffix type
            ax=axes[i],
            edgecolor="black",
            palette="Set1",  # You can choose another palette
        )

        # Optionally draw a baseline line y = x
        if baseline_line:
            min_val = 0.3
            max_val = 1.0
            axes[i].plot(
                [min_val, max_val],
                [min_val, max_val],
                color="black",
                linestyle="--",
                linewidth=2,
                label="y = x",  # Add a label for the line
            )
            axes[i].legend(loc="best")  # Add legend to display the label

        axes[i].set_xlim(x_min, x_max)
        axes[i].set_ylim(y_min, y_max)

        axes[i].set_title(f"{test_set}", fontsize=20)
        axes[i].set_xlabel(iid_label, fontsize=14)
        axes[i].set_ylabel(ood_label, fontsize=14)

        # Move the legend outside if desired, or let seaborn/Matplotlib handle it
        if i == 0:  # For the first subplot, you might position the legend
            axes[i].legend(loc="best")
        else:
            axes[i].legend().remove()

    # Turn off any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_correlation_matrix(
    df, metric_prefix="accuracy", corr_type="kendall", title="Correlation Matrix", save_path=None
):
    """
    Plot a correlation matrix for accuracy metrics in a single DataFrame, excluding "Top-5" metrics.

    Parameters:
    - df: DataFrame containing accuracy metrics (wide format).
    - metric_prefix: Prefix for accuracy columns to include in the correlation (default: "accuracy").
    - title: Title for the correlation matrix plot.
    """
    # Filter accuracy columns based on the prefix, excluding "Top-5"
    columns = [
        col for col in df.columns if metric_prefix.lower() in col.lower() and "top5" not in col.lower()
    ]

    # Compute the correlation matrix
    corr_matrix = df[columns].rename(columns=lambda x: x.replace("_accuracy_gap", "")).corr(corr_type)

    # Plot the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        square=True,
        annot_kws={"size": 20},
        vmin=-1,  # Set minimum value for the color scale
        vmax=1,
    )
    plt.title(title)
    plt.xticks(rotation=45, ha="right", fontsize=15)
    plt.yticks(fontsize=15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_pairplot(
    dataframe,
    hue=None,
    diag_kind="kde",
    palette="viridis",
    plot_size=2.5,
):
    """
    Creates a pair plot for the given dataset.

    Parameters:
    - dataframe (pd.DataFrame): The DataFrame containing the data to visualize.
    - hue (str, optional): Column name for color-coding the points based on categories.
    - diag_kind (str, optional): Kind of plot for the diagonal subplots ('kde' or 'hist'). Default is 'kde'.
    - palette (str, optional): Color palette for the plot. Default is 'viridis'.
    - plot_size (float, optional): Size of each subplot. Default is 2.5.

    Returns:
    - PairGrid object for further customization (if needed).
    """
    # Create the pair plot
    pair_plot = sns.pairplot(
        dataframe,
        hue=hue,
        diag_kind=diag_kind,
        palette=palette,
        height=plot_size,
    )
    plt.show()

    return pair_plot


def dataframe_pairwise_correlation(
    df1, df2, x_label, y_label, method="kendall", title="Correlation Heatmap", save_path=None
):
    """
    Compute and plot the correlation matrix between selected columns of two DataFrames.

    Parameters:
    - df1: First DataFrame.
    - cols_df1: List of columns to use from the first DataFrame.
    - df2: Second DataFrame.
    - cols_df2: List of columns to use from the second DataFrame.
    - method: Correlation method (default: "kendall"). Options: "pearson", "kendall", "kendall".
    - title: Title for the heatmap.
    - save_path: If specified, saves the figure as a PDF to the given path.
    """
    # Subset the DataFrames with the selected columns
    cols_df1 = df1.columns
    cols_df2 = df2.columns

    # Initialize an empty DataFrame for the correlation matrix
    correlation_matrix = pd.DataFrame(index=cols_df1, columns=cols_df2)

    # Compute correlations
    for col_df1 in cols_df1:
        for col_df2 in cols_df2:
            correlation_matrix.loc[col_df1, col_df2] = df1[col_df1].corr(df2[col_df2], method=method)

    # Convert correlation matrix to float for visualization
    correlation_matrix = correlation_matrix.astype(float)

    # Plot the heatmap
    plt.figure(figsize=(13, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        square=True,
        annot_kws={"size": 20},
        vmin=-1,  # Set minimum value for the color scale
        vmax=1,  # Set maximum value for the color scale
    )
    plt.title(title, fontsize=20)
    plt.ylabel(y_label, fontsize=15)
    plt.yticks(rotation=0, fontsize=18)
    plt.xticks(rotation=45, ha="right", fontsize=18)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    plt.show()
