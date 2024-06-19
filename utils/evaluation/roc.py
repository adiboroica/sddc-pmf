import numpy as np
import sparse
from sklearn.metrics import roc_curve, auc


def roc_and_auc(labels: np.ndarray, probabilities: np.ndarray):
    """
    Compute the Receiver Operating Characteristic (ROC) curve and the Area Under the Curve (AUC) value.

    Returns:
    -------
    fpr: np.ndarray
      False Positive Rate
    tpr: np.ndarray
      True Positive Rate
    auc_value: Float
      Area Under the Curve value
    """

    # If any of the input is a sparse matrix, convert it to a dense matrix
    if isinstance(labels, sparse.COO):
        labels = labels.todense()
    if isinstance(probabilities, sparse.COO):
        probabilities = probabilities.todense()

    # Ensure that they have the same shape
    assert labels.shape == probabilities.shape

    # Flatten the input
    labels = labels.flatten()
    probabilities = probabilities.flatten()

    fpr, tpr, _ = roc_curve(labels, probabilities)
    auc_value = auc(fpr, tpr)

    return fpr, tpr, auc_value
