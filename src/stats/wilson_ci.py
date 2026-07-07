"""
Wilson score confidence interval for a binomial proportion (e.g. ASR,
refusal rate). Preferred over the normal-approximation interval, especially
for small n or proportions near 0 or 1 — both common in this dataset
(some categories will have near-0% or near-100% refusal rates for certain
model/language pairs).
"""
from statsmodels.stats.proportion import proportion_confint


def wilson_ci(successes: int, n: int, alpha: float = 0.05):
    """
    Returns (point_estimate, lower_bound, upper_bound).

    Example:
        wilson_ci(successes=42, n=100)  # 42% ASR, 95% CI
    """
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = successes / n
    lower, upper = proportion_confint(successes, n, alpha=alpha, method="wilson")
    return point, lower, upper


if __name__ == "__main__":
    # sanity check
    p, lo, hi = wilson_ci(42, 100)
    print(f"point={p:.3f}  95% CI=({lo:.3f}, {hi:.3f})")
