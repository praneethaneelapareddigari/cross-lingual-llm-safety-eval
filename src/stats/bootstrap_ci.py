"""
Nonparametric bootstrap confidence interval — for any metric without a
closed-form variance formula, e.g. mean harmfulness score, or a composite
"safety index" combining ASR + harmfulness + false-refusal rate.
"""
import numpy as np


def bootstrap_ci(data: list[float], statistic=np.mean, n_boot: int = 10000,
                  alpha: float = 0.05, seed: int = 42):
    """
    Returns (point_estimate, lower_bound, upper_bound) using the percentile
    bootstrap method.

    data: list/array of per-prompt values (e.g. harmfulness scores 0-4)
    statistic: callable applied to each resample (default: mean)
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    n = len(data)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))

    point = statistic(data)
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = statistic(sample)

    lower = np.percentile(boot_stats, 100 * (alpha / 2))
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return point, lower, upper


def bootstrap_diff_ci(group_a: list[float], group_b: list[float],
                       statistic=np.mean, n_boot: int = 10000, alpha: float = 0.05,
                       seed: int = 42):
    """
    Bootstrap CI for the DIFFERENCE in a statistic between two independent
    groups (e.g. mean harmfulness in Tamil minus mean harmfulness in English).
    Useful when McNemar (paired) doesn't apply, e.g. comparing across models
    rather than across languages for the same model.
    """
    rng = np.random.default_rng(seed)
    a, b = np.asarray(group_a), np.asarray(group_b)
    point = statistic(a) - statistic(b)

    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        sample_a = rng.choice(a, size=len(a), replace=True)
        sample_b = rng.choice(b, size=len(b), replace=True)
        boot_diffs[i] = statistic(sample_a) - statistic(sample_b)

    lower = np.percentile(boot_diffs, 100 * (alpha / 2))
    upper = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
    return point, lower, upper


if __name__ == "__main__":
    demo = [0, 1, 0, 2, 3, 0, 1, 4, 0, 2]
    print(bootstrap_ci(demo))
