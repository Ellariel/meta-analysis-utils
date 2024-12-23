import semopy

from metatools.sem import sem, sem_report


def test_sem():
    # https://semopy.com/tutorial.html
    data = semopy.examples.political_democracy.get_data()
    formula = semopy.examples.political_democracy.get_model()
    assert len(data) == 75

    stats, metrics = sem(
        data,
        formula,
        method="MLW",  # MLW ULS GLS FIML DWLS WLS
        solver="SLSQP",
        bootstrap=10,
        se_robust=True,
        standardized=True,
        seed=13,
    )

    assert f"{stats['estimate'].iloc[0]:.3f}" == "1.252"

    stats = sem_report(stats, metrics)
    assert (
        stats.iloc[0]["model"]
        == "χ2(35, N = 75) = 38.125, p = .329, CFI = .995, GFI = .948, AGFI = .918, NFI = .948, TLI = .993, RMSEA = 0.035, AIC = 60.872, BIC = 132.714, LogLik = 0.564"
    )
    print(stats)


if __name__ == "__main__":
    test_sem()
    print("Tests for metatools.sem are PASSED!")
