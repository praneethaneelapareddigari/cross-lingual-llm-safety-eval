"""
Cohen's h — effect size for the difference between two proportions.
Report this ALONGSIDE McNemar's p-value: a statistically significant
difference (small p) can still be a tiny, practically meaningless effect,
especially at your larger sample sizes. h fills that gap.

Conventional interpretation (Cohen, 1988):
    |h| < 0.2   negligible
    0.2 - 0.5   small
    0.5 - 0.8   medium
    > 0.8       large
"""
import numpy as np


def cohens_h(p1: float, p2: float) -> float:
    """
    p1, p2: two proportions (e.g. refusal rate in English vs. refusal rate
    in Telugu), each in [0, 1].
    """
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    return phi1 - phi2


def interpret_h(h: float) -> str:
    ah = abs(h)
    if ah < 0.2:
        return "negligible"
    elif ah < 0.5:
        return "small"
    elif ah < 0.8:
        return "medium"
    else:
        return "large"


if __name__ == "__main__":
    h = cohens_h(0.85, 0.55)  # e.g. 85% refusal in English vs 55% in Telugu
    print(f"Cohen's h = {h:.3f} ({interpret_h(h)} effect)")
