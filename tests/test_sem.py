import numpy as np
import statsmodels.api as sm

from metatools.sem import sem


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

    stats, metrics, fig = sem(
        test_data,
        formula,
        method="MLW",  # MLW ULS GLS FIML DWLS WLS
        solver="SLSQP",
        bootstrap=10,
        plot_covs=True,
        se_robust=False,
        standardized=True,
        format_pval=True,
        save_to_file="fig.pdf",
        return_fig=True,
        decimal=3,
        seed=13,
    )
    
    print(metrics)
    print(stats)
    print(fig)


if __name__ == "__main__":
    test_sem()
    print("Tests for metatools.sem are PASSED!")
