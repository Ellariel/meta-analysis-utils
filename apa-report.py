import math, random, scipy
import numpy as np

from scipy.stats import wilcoxon, mannwhitneyu
from scipy.stats import pearsonr, spearmanr

######################################################################################

def errorbars(data, method='std'):
    if method == 'std':
        y = np.std(data)
    return y

def z_from_p(p, method='two-tailed'):
    # https://www.gigacalculator.com/calculators/p-value-to-z-score-calculator.php
    z = scipy.stats.norm.ppf(p/2) if method == 'two-tailed' else scipy.stats.norm.ppf(p)
    return -z

def bootstrap_CI(data, func=np.mean, p=0.95, n=1000, seed=13):
    # Bootstrapped 95% confidence intervals for the mean/median value from 1000 resamples are reported.
    # https://towardsdatascience.com/how-to-calculate-confidence-intervals-in-python-a8625a48e62b
    random.seed(seed)
    np.random.seed(seed)
    simulations = [func(np.random.choice(data, size=len(data), replace=True)) for i in range(n)]
    lp, rp, m = (1-p)/2, 1-(1-p)/2, func(data)
    return np.hstack([m, np.percentile(simulations, [lp*100, rp*100])])

def spearman_CI(r, n, p=0.95):
    # https://en.wikipedia.org/wiki/Fisher_transformation
    # https://stats.stackexchange.com/questions/18887/how-to-calculate-a-confidence-interval-for-spearmans-rank-correlation
    se = 1 / math.sqrt(n-3)
    d = z_from_p(1-p) * se
    return np.array([math.tanh(math.atanh(r) - d), math.tanh(math.atanh(r) + d)])

def cohen_r_from_p(p, n, method='two-tailed'):
    # Cohen’s guidelines for r are that a large effect is .5, a medium effect is .3, and a small effect is .1 (Coolican, 2009, p. 395).
    # Fritz, C. O., Morris, P. E., & Richler, J. J. (2012). Effect size estimates: Current use, calculations, and interpretation. Journal of Experimental Psychology: General, 141(1), 2–18. https://doi.org/10.1037/a0024338
    # https://www.researchgate.net/publication/51554230_Effect_Size_Estimates_Current_Use_Calculations_and_Interpretation
    return z_from_p(p, method=method) / math.sqrt(n)
    
def cohen_r_from_z(z, n):
    # Cohen’s guidelines for r are that a large effect is .5, a medium effect is .3, and a small effect is .1 (Coolican, 2009, p. 395).
    # Fritz, C. O., Morris, P. E., & Richler, J. J. (2012). Effect size estimates: Current use, calculations, and interpretation. Journal of Experimental Psychology: General, 141(1), 2–18. https://doi.org/10.1037/a0024338
    # https://www.researchgate.net/publication/51554230_Effect_Size_Estimates_Current_Use_Calculations_and_Interpretation
    return z / math.sqrt(n)

def cohen_d_from_f(f, df_num, df_denom):
    # Cohen’s d derived from F-statistic (Fritz et al., 2012)
    # Fritz, C. O., Morris, P. E., & Richler, J. J. (2012). Effect size estimates: Current use, calculations, and interpretation. Journal of Experimental Psychology: General, 141(1), 2–18. https://doi.org/10.1037/a0024338
    # https://www.researchgate.net/publication/51554230_Effect_Size_Estimates_Current_Use_Calculations_and_Interpretation
    # https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00863/full
    eta_sq = (f * df_num) / (f * df_num + df_denom)
    d = 2 * math.sqrt(eta_sq) / math.sqrt(1 - eta_sq)
    return d
