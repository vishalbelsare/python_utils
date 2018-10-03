import numpy as np
import warnings


def neg_brier_loss_pointwise(y_true, y_pred):
    return -(y_true - y_pred[:, 1]) ** 2


def accuracy_score_pointwise(y_true, y_pred):
    return y_true == y_pred


def neg_log_loss_pointwise(y_true, y_pred, eps=1e-15):
    y_pred_clipped = np.clip(y_pred[:, 1], eps, 1 - eps)
    loss = -((y_true * np.log(y_pred_clipped)) + (1 - y_true)
             * np.log(1 - y_pred_clipped))
    return -loss


def specificity_score(y_true, y_pred):
    tn = np.logical_and(y_true == 0, y_pred == 0)
    return _divide(np.sum(tn), np.sum(y_true == 0))


def sensitivity_score(y_true, y_pred):
    tp = np.logical_and(y_true == 1, y_pred == 1)
    return _divide(np.sum(tp), np.sum(y_true))


def precision_score(y_true, y_pred):
    tp = np.logical_and(y_true == 1, y_pred == 1)
    return _divide(np.sum(tp), np.sum(y_pred))


def _divide(numerator, denominator):
    # similar to sklearn.metrics.classification._prf_divide
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / denominator
    if denominator == 0.0:
        warnings.warn('Metric ill-defined and set to 0.0')
        result = 0.0
    return result
