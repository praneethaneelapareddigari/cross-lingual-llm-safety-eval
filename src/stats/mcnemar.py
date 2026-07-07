"""
McNemar's test for paired binary outcomes — the core test for "did this model
refuse the SAME prompt differently in English vs. language X?"

Requires prompts to be paired 1:1 across languages (same underlying seed
prompt, translated). Build the 2x2 contingency table:

                        Target-language: REFUSED   Target-language: NOT REFUSED
    English: REFUSED           a                           b
    English: NOT REFUSED       c                           d

McNemar's test only uses the discordant pairs (b, c) — cases where behavior
*differed* between languages. This is the right test here (not a
chi-square/independent-samples test) because the same prompt content is
compared across two conditions on the same model.
"""
from statsmodels.stats.contingency_tables import mcnemar


def mcnemar_test(a: int, b: int, c: int, d: int, exact: bool = True):
    """
    a = both refused
    b = refused in English only
    c = refused in target language only
    d = neither refused

    Returns statsmodels result object with .pvalue and .statistic.
    Uses exact binomial test by default (recommended when b+c < 25); set
    exact=False to use chi-square with continuity correction for larger n.
    """
    table = [[a, b], [c, d]]
    result = mcnemar(table, exact=exact, correction=not exact)
    return result


def build_pairs_and_test(pairs: list[tuple[bool, bool]], exact: bool = None):
    """
    pairs: list of (refused_in_english: bool, refused_in_target_lang: bool)
    for the SAME underlying prompt, same model.

    Automatically picks exact vs. chi-square based on discordant pair count.
    """
    a = sum(1 for en, tgt in pairs if en and tgt)
    b = sum(1 for en, tgt in pairs if en and not tgt)
    c = sum(1 for en, tgt in pairs if not en and tgt)
    d = sum(1 for en, tgt in pairs if not en and not tgt)

    discordant = b + c
    use_exact = exact if exact is not None else (discordant < 25)

    result = mcnemar_test(a, b, c, d, exact=use_exact)
    return {
        "a_both_refused": a,
        "b_refused_en_only": b,
        "c_refused_target_only": c,
        "d_neither_refused": d,
        "discordant_pairs": discordant,
        "test_used": "exact_binomial" if use_exact else "chi_square_corrected",
        "statistic": result.statistic,
        "pvalue": result.pvalue,
    }


if __name__ == "__main__":
    # sanity check: refused in English but not in Telugu for 8 prompts (asymmetry)
    demo_pairs = [(True, False)] * 8 + [(True, True)] * 40 + [(False, False)] * 45 + [(False, True)] * 2
    print(build_pairs_and_test(demo_pairs))
