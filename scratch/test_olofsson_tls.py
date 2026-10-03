from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("f:/gis-project/UrbanPulse")
DATA_DIR = PROJECT_ROOT / "data"

PROJECT_CLASS_NAMES = {
    1: "Built-up",
    2: "Vegetation",
    3: "Water",
    4: "Agriculture",
    5: "Open land",
}
FEATURE_NAMES = ["red", "green", "blue", "nir", "swir16", "ndvi", "ndbi", "mndwi"]
OPTICAL_BANDS = ["blue", "green", "red", "nir", "swir16"]
YEARS = list(range(2018, 2025))
REF_YEAR = 2021

# Orthogonal / Total Least Squares (TLS) function
def fit_tls_regression(x, y):
    """
    Fits Orthogonal Distance Regression (Total Least Squares / Major Axis):
    y = m * x + c
    Minimizes perpendicular squared distances from points to the line.
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_c = x - x_mean
    y_c = y - y_mean

    # 2x2 covariance / scatter matrix SVD
    M = np.column_stack([x_c, y_c])
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    # First principal component vector
    v1 = Vt[0]  # (vx, vy)
    if v1[0] == 0:
        slope = 1.0
    else:
        slope = float(v1[1] / v1[0])
    intercept = float(y_mean - slope * x_mean)

    # Correlation coefficient r
    r = np.corrcoef(x, y)[0, 1]
    return slope, intercept, float(r**2)

def safe_norm_diff(a, b):
    denom = a + b
    valid = (np.abs(denom) > 1e-5) & np.isfinite(a) & np.isfinite(b)
    res = np.full_like(a, np.nan, dtype=np.float32)
    np.divide(a - b, denom, out=res, where=valid)
    return np.clip(res, -1.0, 1.0)

def apply_majority_filter_3x3(arr, valid_mask):
    h, w = arr.shape
    padded = np.pad(arr, 1, mode="edge")
    neighbors = np.stack([
        padded[0:h, 0:w], padded[0:h, 1:w+1], padded[0:h, 2:w+2],
        padded[1:h+1, 0:w], padded[1:h+1, 1:w+1], padded[1:h+1, 2:w+2],
        padded[2:h+2, 0:w], padded[2:h+2, 1:w+1], padded[2:h+2, 2:w+2],
    ], axis=0)
    class_votes = np.stack([(neighbors == c).sum(axis=0) for c in range(1, 6)], axis=0)
    majority_class = (np.argmax(class_votes, axis=0) + 1).astype(np.uint8)
    return np.where(valid_mask, majority_class, 0).astype(np.uint8)

# Olofsson et al. 2014 Area Estimation
def olofsson_area_estimation(cm, mapped_area_km2_by_class, target_class_idx=0):
    """
    Computes area-weighted stratified estimate of area and 95% CI according to Olofsson et al. (2014).
    cm: Confusion matrix of shape (K, K), where rows=mapped class i (1..K), cols=reference class j (1..K).
    mapped_area_km2_by_class: Array of length K containing mapped area in km2 for each class i.
    target_class_idx: 0 for Built-up (class 1).
    """
    K = cm.shape[0]
    A_total = np.sum(mapped_area_km2_by_class)
    W = mapped_area_km2_by_class / A_total  # Mapped area proportions W_i

    n_i_dot = np.sum(cm, axis=1)  # row sums (mapped sample counts)

    # Stratified area proportion matrix p_ij = W_i * (n_ij / n_i_dot)
    p = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        if n_i_dot[i] > 0:
            p[i, :] = W[i] * (cm[i, :] / n_i_dot[i])

    # Estimated reference class area proportion p_dot_k
    p_dot_k = np.sum(p[:, target_class_idx])
    A_adj_km2 = A_total * p_dot_k

    # Standard error of p_dot_k
    var_p_dot_k = 0.0
    for i in range(K):
        if n_i_dot[i] > 1:
            n_ik = cm[i, target_class_idx]
            n_i = n_i_dot[i]
            sample_prop = n_ik / n_i
            var_term = (W[i]**2) * (sample_prop * (1.0 - sample_prop)) / (n_i - 1)
            var_p_dot_k += var_term

    se_p_dot_k = np.sqrt(var_p_dot_k)
    se_A_adj_km2 = A_total * se_p_dot_k
    ci_95_km2 = 1.96 * se_A_adj_km2

    mapped_target_km2 = mapped_area_km2_by_class[target_class_idx]

    return {
        "mapped_area_km2": mapped_target_km2,
        "adjusted_area_km2": A_adj_km2,
        "se_km2": se_A_adj_km2,
        "ci_95_km2": ci_95_km2,
        "ci_lower_km2": A_adj_km2 - ci_95_km2,
        "ci_upper_km2": A_adj_km2 + ci_95_km2,
    }

print("Loaded definitions successfully.")
