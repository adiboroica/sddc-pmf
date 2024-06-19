import os
import numpy as np
import matplotlib.pyplot as plt

from pmf import PMFModel
from utils import evaluation
from utils import evaluation


pictures_folder_path = "pictures"

num_train = 10
num_test = 4


def plot_eval(model: PMFModel, type: str, color):

    eval_indices = model.cavi_results.eval_indices

    # Choose the appropriate data
    match type:
        case "elbo":
            eval_values = model.cavi_results.elbo_values
            label = "ELBO"
        case "log-likelihood":
            eval_values = model.cavi_results.log_likelihood_values
            label = "Log Likelihood"
        case _:
            raise ValueError("Invalid type")

    plt.figure(figsize=(6, 3), dpi=300)

    plt.plot(
        eval_indices,
        eval_values,
        label=label,
        color=color,
        linestyle="-",
        linewidth=2,
    )

    plt.title(f"{label} Values during training")
    plt.xlabel("Iteration")
    plt.ylabel(f"{label}")
    plt.legend()
    plt.grid(False)

    plt.savefig(
        os.path.join(pictures_folder_path, f"{type}.pdf"),
        format="pdf",
        bbox_inches="tight",
    )

    plt.show()


def sample_indices_test_data(test_dataset, print_num_samples=True):
    test_indices_axis0 = []
    test_indices_axis1 = []
    for time_step in range(num_test):
        print(f"Time step {time_step + 1}")

        num_negative_indices = np.sum(test_dataset[time_step] == 0)

        threshold = 5000
        if num_negative_indices <= threshold:
            indices_axis0, indices_axis1, num_positives, num_negatives = (
                evaluation.sample(
                    test_dataset[time_step],
                    return_num_every_class=True,
                    rng=np.random.default_rng(43),
                )
            )
        else:
            indices_axis0, indices_axis1, num_positives, num_negatives = (
                evaluation.sample(
                    test_dataset[time_step],
                    num_negative_labels=threshold,
                    return_num_every_class=True,
                    rng=np.random.default_rng(43),
                )
            )

        if print_num_samples:
            print(
                f"total: {num_positives + num_negatives}; "
                + f"positive: {num_positives}; "
                + f"negative: {num_negatives}"
            )

        test_indices_axis0.append(indices_axis0)
        test_indices_axis1.append(indices_axis1)

    return test_indices_axis0, test_indices_axis1


def compute_model_predictions(model, indices_axis0, indices_axis1):
    model_predictions = []
    for time_step in range(num_test):
        predictions = model.predict(
            indices_axis0[time_step],
            indices_axis1[time_step],
            n_periods=time_step + 1,
        )
        if time_step > 0:
            predictions = predictions[time_step]
        model_predictions.append(predictions)

    return model_predictions


def compute_fpr_tpr_auc_aip(
    test_dataset,
    test_dataset_indices_axis0,
    test_dataset_indices_axis1,
    aip_all_predictions,
):
    fpr_values = []
    tpr_values = []
    auc_values = np.zeros(num_test)

    for time_step in range(num_test):
        # Extract the indices
        idx_axis0 = test_dataset_indices_axis0[time_step]
        idx_axis1 = test_dataset_indices_axis1[time_step]

        # Extract the test data
        true_labels = test_dataset[time_step][idx_axis0, idx_axis1]

        # Extract the predictions
        aip_predictions = aip_all_predictions[time_step][idx_axis0, idx_axis1]

        # Compute the FPR, TPR, and AUC
        fpr, tpr, auc_values[time_step] = evaluation.roc_and_auc(
            true_labels, aip_predictions
        )

        fpr_values.append(fpr)
        tpr_values.append(tpr)

    return fpr_values, tpr_values, auc_values


def compute_fpr_tpr_auc_cosie(
    test_dataset,
    test_dataset_indices_axis0,
    test_dataset_indices_axis1,
    cosie_all_predictions,
):
    fpr_values = []
    tpr_values = []
    auc_values = np.zeros(num_test)

    for time_step in range(num_test):
        # Extract the indices
        idx_axis0 = test_dataset_indices_axis0[time_step]
        idx_axis1 = test_dataset_indices_axis1[time_step]

        # Extract the test data
        true_labels = test_dataset[time_step][idx_axis0, idx_axis1]

        # Extract the predictions
        cosie_predictions = cosie_all_predictions[time_step][idx_axis0, idx_axis1]

        # Compute the FPR, TPR, and AUC

        fpr, tpr, auc_values[time_step] = evaluation.roc_and_auc(
            true_labels, cosie_predictions
        )

        fpr_values.append(fpr)
        tpr_values.append(tpr)

    return fpr_values, tpr_values, auc_values


def compute_fpr_tpr_auc_model(
    model_predictions,
    test_dataset,
    test_dataset_indices_axis0,
    test_dataset_indices_axis1,
):
    fpr_values = []
    tpr_values = []
    auc_values = np.zeros(num_test)

    for time_step in range(num_test):
        # Extract the indices
        idx_axis0 = test_dataset_indices_axis0[time_step]
        idx_axis1 = test_dataset_indices_axis1[time_step]

        # Extract the test data
        true_labels = test_dataset[time_step][idx_axis0, idx_axis1]

        # Extract the predictions
        model_predictions_current = model_predictions[time_step]

        # Compute the FPR, TPR, and AUC
        fpr, tpr, auc_values[time_step] = evaluation.roc_and_auc(
            true_labels, model_predictions_current
        )

        fpr_values.append(fpr)
        tpr_values.append(tpr)

    return fpr_values, tpr_values, auc_values


def plot_auc_values(
    range,
    model_auc_values,
    cosie_auc_values,
    aip_auc_values,
    title: str,
    xlabel: str,
    file_name: str,
):
    # Use a color-blind friendly palette
    colors = plt.get_cmap("tab10").colors

    plt.figure(figsize=(6, 3), dpi=300)

    plt.plot(
        range,
        aip_auc_values,
        label="AIP",
        color=colors[2],
        linestyle="-.",
        linewidth=1,
        marker="o",
        markersize=5,
    )
    plt.plot(
        range,
        cosie_auc_values,
        label="COSIE",
        color=colors[1],
        linestyle="--",
        linewidth=1,
        marker="o",
        markersize=5,
    )
    plt.plot(
        range,
        model_auc_values,
        label="Proposed Model",
        color=colors[0],
        linestyle="-",
        linewidth=1,
        marker="o",
        markersize=5,
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.xticks(range)
    plt.ylabel("AUC")
    plt.legend()
    plt.grid(False)

    plt.savefig(
        os.path.join(pictures_folder_path, file_name),
        format="pdf",
        bbox_inches="tight",
    )

    plt.show()


def plot_diff_auc_values(
    activity_levels,
    first_auc_values,
    second_auc_values,
    models: str,
    type: str,
    file_name: str = None,
):
    # Use a color-blind friendly palette
    colors = plt.get_cmap("tab10").colors
    linestyles = ["--", "-.", ":", "--"]

    plt.figure(figsize=(6, 3), dpi=300)

    for i in range(num_test):
        # Check for invalid values and handle them
        valid_indices = ~np.isnan(first_auc_values[i])

        plt.plot(
            activity_levels[valid_indices],
            first_auc_values[i][valid_indices] - second_auc_values[i][valid_indices],
            label=f"t = {num_train + i + 1}",
            color=colors[i],
            linestyle=linestyles[i],
            linewidth=1,
            marker="o",
            markersize=5,
        )

    # Draw a line at y = 0
    plt.axhline(y=0, color="black", linestyle="-", linewidth=2)

    plt.title(f"Difference in AUC Values: {models}")
    plt.xlabel(f"{type} Activity Level")
    plt.ylabel("Difference in AUC")
    plt.xticks(activity_levels)
    plt.legend()
    plt.grid(False)

    if file_name:
        plt.savefig(
            os.path.join(pictures_folder_path, file_name),
            format="pdf",
            bbox_inches="tight",
        )

    plt.show()
