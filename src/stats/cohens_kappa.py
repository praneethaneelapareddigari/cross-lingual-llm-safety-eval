"""
Cohen's Kappa — inter-annotator agreement, used for:
  (1) LLM-judge vs. human annotator agreement, on the ~10% human-reviewed
      subsample (validates that the judge pipeline is trustworthy at scale)
  (2) Human annotator vs. human annotator agreement (if you have 2+ reviewers
      on the self-harm / sensitive categories)

Report this number in the paper's methods section — reviewers will want to
know the judge is reliable before they trust ASR numbers computed from it.
"""
from sklearn.metrics import cohen_kappa_score

# Landis & Koch (1977) interpretation bands, commonly cited
INTERPRETATION_BANDS = [
    (0.00, "poor"),
    (0.20, "slight"),
    (0.40, "fair"),
    (0.60, "moderate"),
    (0.80, "substantial"),
    (1.01, "almost perfect"),
]


def compute_kappa(labels_a: list, labels_b: list) -> float:
    """
    labels_a, labels_b: parallel lists of categorical labels (e.g.
    ["REFUSAL", "FULL_COMPLY", ...]) from two raters over the SAME items.
    """
    assert len(labels_a) == len(labels_b), "Rater label lists must be same length"
    return cohen_kappa_score(labels_a, labels_b)


def interpret_kappa(k: float) -> str:
    for threshold, label in INTERPRETATION_BANDS:
        if k < threshold:
            return label
    return INTERPRETATION_BANDS[-1][1]


if __name__ == "__main__":
    human = ["REFUSAL", "FULL_COMPLY", "PARTIAL_COMPLY", "REFUSAL", "FULL_COMPLY"]
    judge = ["REFUSAL", "FULL_COMPLY", "REFUSAL", "REFUSAL", "FULL_COMPLY"]
    k = compute_kappa(human, judge)
    print(f"Cohen's Kappa = {k:.3f} ({interpret_kappa(k)} agreement)")
