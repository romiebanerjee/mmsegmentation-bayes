# import time, json
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import pickle, os
# import math

# from scipy.stats import gmean
# from matplotlib.patches import Ellipse
# import matplotlib.transforms as transforms
# from sklearn.covariance import MinCovDet


# def probabilities(UNCERTAINTY_THRESHOLD, ERROR_THRESHOLD, data_np):
#     N,_ = np.shape(data_np)
#     corrects = data_np[:,0] < ERROR_THRESHOLD
#     incorrects = data_np[:,0] >= ERROR_THRESHOLD

#     uncertain_indices = data_np[:, 1] >= UNCERTAINTY_THRESHOLD 
#     uncertain_data = data_np[uncertain_indices, :] #samples which are uncertain
#     uncertain_and_correct = uncertain_data[:,0] < ERROR_THRESHOLD
#     uncertain_and_incorrect = uncertain_data[:,0] >= ERROR_THRESHOLD

#     certain_indices = data_np[:, 1] < UNCERTAINTY_THRESHOLD 
#     certain_data = data_np[certain_indices, :] #samples which are certain
#     certain_and_correct = certain_data[:,0] < ERROR_THRESHOLD
#     certain_and_incorrect = certain_data[:,0] >= ERROR_THRESHOLD

#     uncertain_given_incorrect_prob = np.sum(uncertain_and_incorrect)/np.sum(incorrects)
#     uncertain_given_correct_prob = np.sum(uncertain_and_correct)/np.sum(corrects)
#     correct_given_certain_prob = np.sum(certain_and_correct)/np.sum(certain_indices)
#     AvU = (np.sum(certain_and_correct) + np.sum(uncertain_and_incorrect))/N


#     return AvU, uncertain_given_correct_prob, uncertain_given_incorrect_prob, correct_given_certain_prob


import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt


def calculate_avu_metrics(
    uncertainty_threshold: float,
    error_threshold: float,
    data_np: np.ndarray,
    return_counts: bool = False
) -> Tuple[float, float, float, float]:
    """
    Calculate Accuracy vs Uncertainty (AvU) metrics and related probabilities.
    
    Args:
        uncertainty_threshold: Threshold to separate certain/uncertain predictions
        error_threshold: Threshold to separate correct/incorrect predictions
        data_np: Numpy array with shape (N, 2) where:
                 - data_np[:, 0] contains error values
                 - data_np[:, 1] contains uncertainty values
        return_counts: If True, returns raw counts instead of probabilities
    
    Returns:
        Tuple containing:
        - AvU score (Accuracy vs Uncertainty)
        - P(Uncertain|Correct)
        - P(Uncertain|Incorrect)
        - P(Correct|Certain)
        If return_counts=True, returns raw counts instead of probabilities
    """
    # Input validation
    assert data_np.ndim == 2 and data_np.shape[1] == 2, "Input must be (N, 2) array"
    assert np.all(data_np[:, 1] >= 0), "Uncertainty values must be non-negative"
    
    N = data_np.shape[0]
    
    # Calculate basic categories
    correct = data_np[:, 0] < error_threshold
    incorrect = ~correct
    
    uncertain = data_np[:, 1] >= uncertainty_threshold
    certain = ~uncertain
    
    # Calculate intersections
    uncertain_correct = uncertain & correct
    uncertain_incorrect = uncertain & incorrect
    certain_correct = certain & correct
    certain_incorrect = certain & incorrect
    
    # Calculate metrics
    if return_counts:
        AvU = (np.sum(certain_correct) + np.sum(uncertain_incorrect))
        return (
            AvU,
            np.sum(uncertain_correct),
            np.sum(uncertain_incorrect),
            np.sum(certain_correct)
        )
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            uncertain_given_correct = np.sum(uncertain_correct) / np.sum(correct)
            uncertain_given_incorrect = np.sum(uncertain_incorrect) / np.sum(incorrect)
            correct_given_certain = np.sum(certain_correct) / np.sum(certain)
            AvU = (np.sum(certain_correct) + np.sum(uncertain_incorrect)) / N
        
        # Handle cases with zero division
        uncertain_given_correct = 0.0 if np.isnan(uncertain_given_correct) else uncertain_given_correct
        uncertain_given_incorrect = 0.0 if np.isnan(uncertain_given_incorrect) else uncertain_given_incorrect
        correct_given_certain = 0.0 if np.isnan(correct_given_certain) else correct_given_certain
        
        return AvU, uncertain_given_correct, uncertain_given_incorrect, correct_given_certain


def find_optimal_thresholds(
    data_np: np.ndarray,
    uncertainty_range: np.ndarray = None,
    error_range: np.ndarray = None,
    n_points: int = 100
) -> Tuple[float, float, float]:
    """
    Find thresholds that maximize the AvU score.
    
    Args:
        data_np: Input data array (N, 2)
        uncertainty_range: Range of uncertainty values to test
        error_range: Range of error values to test
        n_points: Number of points to test in each dimension
    
    Returns:
        Tuple of (best_AvU, best_uncertainty_threshold, best_error_threshold)
    """
    if uncertainty_range is None:
        uncertainty_range = np.linspace(data_np[:, 1].min(), data_np[:, 1].max(), n_points)
    if error_range is None:
        error_range = np.linspace(data_np[:, 0].min(), data_np[:, 0].max(), n_points)
    
    best_AvU = -1
    best_unc_thresh = 0
    best_err_thresh = 0
    
    for unc_thresh in uncertainty_range:
        for err_thresh in error_range:
            AvU, _, _, _ = calculate_avu_metrics(unc_thresh, err_thresh, data_np)
            if AvU > best_AvU:
                best_AvU = AvU
                best_unc_thresh = unc_thresh
                best_err_thresh = err_thresh
                
    return best_AvU, best_unc_thresh, best_err_thresh


def plot_avu_analysis(
    data_np: np.ndarray,
    uncertainty_threshold: float = None,
    error_threshold: float = None,
    save_path: str = None
):
    """
    Visualize the AvU analysis with scatter plots and metric displays.
    
    Args:
        data_np: Input data array (N, 2)
        uncertainty_threshold: Threshold for uncertainty
        error_threshold: Threshold for error
        save_path: If provided, saves the plot to this path
    """
    if uncertainty_threshold is None or error_threshold is None:
        _, uncertainty_threshold, error_threshold = find_optimal_thresholds(data_np)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot with decision boundaries
    sc = ax[0].scatter(data_np[:, 1], data_np[:, 0], alpha=0.5)
    ax[0].axhline(y=error_threshold, color='r', linestyle='--')
    ax[0].axvline(x=uncertainty_threshold, color='r', linestyle='--')
    ax[0].set_xlabel('Uncertainty')
    ax[0].set_ylabel('Error')
    ax[0].set_title('Error vs Uncertainty')
    
    # Calculate metrics
    AvU, p_unc_correct, p_unc_incorrect, p_correct_certain = calculate_avu_metrics(
        uncertainty_threshold, error_threshold, data_np
    )
    
    # Metrics display
    metrics_text = (
        f"AvU Score: {AvU:.3f}\n"
        f"P(Uncertain|Correct): {p_unc_correct:.3f}\n"
        f"P(Uncertain|Incorrect): {p_unc_incorrect:.3f}\n"
        f"P(Correct|Certain): {p_correct_certain:.3f}\n"
        f"Uncertainty Threshold: {uncertainty_threshold:.3f}\n"
        f"Error Threshold: {error_threshold:.3f}"
    )
    
    ax[1].text(0.1, 0.5, metrics_text, fontsize=12)
    ax[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()