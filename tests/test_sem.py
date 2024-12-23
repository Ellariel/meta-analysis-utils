import numpy as np
import statsmodels.api as sm

from metatools.sem import sem, sem_report


def s(x):
    return f"{x:.2f}"


def test_sem():
    test_data = sm.datasets.get_rdataset("mtcars", "datasets", cache=True).data
    assert len(test_data) == 32
    test_data = test_data[["mpg", "wt", "vs"]]

    formula = """
    w =~ wt
    v =~ vs
    mpg ~ w + v
    """

    stats, metrics = sem(
        test_data,
        formula,
        method="MLW",  # MLW ULS GLS FIML DWLS WLS
        solver="SLSQP",
        bootstrap=10,
        se_robust=True,
        standardized=True,
        seed=13,
    )
    
    print(metrics)
    print(stats)

    stats, metrics = sem_report(stats, metrics)
    
    print(metrics)
    print(stats)

if __name__ == "__main__":
    test_sem()
    print("Tests for metatools.sem are PASSED!")
