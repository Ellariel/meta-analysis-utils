import semopy

from metatools.sem import sem, sem_report


def s(x):
    return f"{x:.3f}"


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

    stats = sem_report(stats, metrics)

    print(stats)


if __name__ == "__main__":
    test_sem()
    print("Tests for metatools.sem are PASSED!")
