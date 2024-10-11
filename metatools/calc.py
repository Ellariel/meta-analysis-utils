import numpy as np
import math
import random
import scipy



def CI_for_m(data, func=np.mean, p=0.95, n=1000, seed=13):
    '''
    Bootstraping confidence intervals for the mean/median value
    
    https://towardsdatascience.com/how-to-calculate-confidence-intervals-in-python-a8625a48e62b
    '''

    random.seed(seed)
    np.random.seed(seed)
    simulations = [func(np.random.choice(data, size=len(data), replace=True)) for i in range(n)]
    lp, rp, m = (1-p)/2, 1-(1-p)/2, func(data)
    return np.hstack([m, np.percentile(simulations, [lp*100, rp*100])])


def CI_for_r(r, n, p=0.95):
    '''
    Bootstraping confidence intervals for the correlation coefficient
    using Fisher transformation
    
    https://stats.stackexchange.com/questions/18887/how-to-calculate-a-confidence-interval-for-spearmans-rank-correlation
    https://en.wikipedia.org/wiki/Fisher_transformation
    '''

    se = 1 / math.sqrt(n-3)
    d = z_from_p(1-p) * se
    return np.array([math.tanh(math.atanh(r) - d), math.tanh(math.atanh(r) + d)])


def t_from_b(b, se): 
    # t-statistic from beta-coefficient
    return b/se


def t_from_z(z): 
    # t-statistic from Fisher's z
    # https://web.cortland.edu/andersmd/STATS/stdscore.html#:~:text=As%20evidenced%20above%2C%20zscores%20are,a%20T%20score%20of%2025.
    return 10*z + 50


def z_from_p(p, method='two-tailed'): 
    # Fisher's z from p-value
    # https://www.gigacalculator.com/calculators/p-value-to-z-score-calculator.php
    z = scipy.stats.norm.ppf(p/2) if method == 'two-tailed' else scipy.stats.norm.ppf(p)
    return -z


def cohen_r_from_p(p, n, method='two-tailed'): 
    # Cohen’s r from p-value
    # Cohen’s guidelines for r are that a large effect is .5, a medium effect is .3, and a small effect is .1 (Coolican, 2009, p. 395).
    # Fritz, C. O., Morris, P. E., & Richler, J. J. (2012). Effect size estimates: Current use, calculations, and interpretation. Journal of Experimental Psychology: General, 141(1), 2–18. https://doi.org/10.1037/a0024338
    # https://www.researchgate.net/publication/51554230_Effect_Size_Estimates_Current_Use_Calculations_and_Interpretation
    return z_from_p(p, method=method) / math.sqrt(n)
    

def cohen_r_from_z(z, n): 
    # Cohen’s r from Fisher's z 
    # Cohen’s guidelines for r are that a large effect is .5, a medium effect is .3, and a small effect is .1 (Coolican, 2009, p. 395).
    # Fritz, C. O., Morris, P. E., & Richler, J. J. (2012). Effect size estimates: Current use, calculations, and interpretation. Journal of Experimental Psychology: General, 141(1), 2–18. https://doi.org/10.1037/a0024338
    # https://www.researchgate.net/publication/51554230_Effect_Size_Estimates_Current_Use_Calculations_and_Interpretation
    return z / math.sqrt(n)


def cohen_d_from_z(z, n): 
    # Cohen’s d from Fisher's z 
    # https://easystats.github.io/effectsize/reference/t_to_r.html
    return 2*z / math.sqrt(n)


def cohen_d_from_t(t, n): 
    # Cohen’s d from t-statistic
    # Fritz, C. O., Morris, P. E., & Richler, J. J. (2012). Effect size estimates: Current use, calculations, and interpretation. Journal of Experimental Psychology: General, 141(1), 2–18. https://doi.org/10.1037/a0024338
    # https://www.researchgate.net/publication/51554230_Effect_Size_Estimates_Current_Use_Calculations_and_Interpretation
    return 2*t / math.sqrt(n-2)


def cohen_d_from_f(f, df_num, df_denom): 
    # Cohen’s d from F-statistic
    # Cohen’s d derived from F-statistic (Fritz et al., 2012)
    # Fritz, C. O., Morris, P. E., & Richler, J. J. (2012). Effect size estimates: Current use, calculations, and interpretation. Journal of Experimental Psychology: General, 141(1), 2–18. https://doi.org/10.1037/a0024338
    # https://www.researchgate.net/publication/51554230_Effect_Size_Estimates_Current_Use_Calculations_and_Interpretation
    # https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00863/full
    eta_sq = (f * df_num) / (f * df_num + df_denom)
    d = 2 * math.sqrt(eta_sq) / math.sqrt(1 - eta_sq)
    return d


def cohen_d_from_r(r): 
    # Cohen's d from correlation r  
    # https://easystats.github.io/effectsize/reference/d_to_r.html
    return math.sqrt(4) * r / math.sqrt(1 - r*r) # assume equal groups


def z_from_r(r): 
    # Fisher's z from correlation r
    # https://www.escal.site/
    # https://en.wikipedia.org/wiki/Fisher_transformation
    return math.atanh(r) # np.log((1 + r)/(1 - r)) / 2


def r_from_cohen_d(d): 
    # Сorrelation r from Cohen's d
    # https://www.escal.site/
    # https://easystats.github.io/effectsize/reference/d_to_r.html
    return d / math.sqrt(d*d + 4) # assume equal groups


def r_from_f(f, df): 
    # Сorrelation r from F-statistic
    # https://juls-dotcom.github.io/meta_analysis.html
    return f / math.sqrt(f + df)
    

def r_from_t(t, df): 
    # Сorrelation r from t-statistic
    # https://juls-dotcom.github.io/meta_analysis.html
    return t*t / math.sqrt(t*t + df)


def r_from_z(z, n): 
    # Сorrelation r (Cohen’s r) from Fisher's z 
    # https://juls-dotcom.github.io/meta_analysis.html
    return z / math.sqrt(n)


def r_from_p(p, n, method='two-tailed'):
    # Сorrelation r (Cohen’s r) from p-value
    return cohen_r_from_p(p, n, method=method)


def unbiased_z(z): 
    # unbiased Z
    # https://juls-dotcom.github.io/meta_analysis.html
    # sample sizes < 20 or 10 in each group, see Nakagawa & Cuthill, 2007
    # https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1469-185X.2007.00027.x
    # (zru) value using the equation proposed by Hedges & Olkin, 1985
    # https://www.sciencedirect.com/book/9780080570655/statistical-methods-for-meta-analysis
    return z


def logit_from_probit(x):
    # https://statmodeling.stat.columbia.edu/2006/06/06/take_logit_coef/
    # https://www.statalist.org/forums/forum/general-stata-discussion/general/1704819-calculating-odds-ration-in-probit-model
    return -1.7*x


def OR_from_logit(x): 
    # odds ratio from log odds ratio
    # odds = EXP(0.873) = 2.394;
    # https://stats.oarc.ucla.edu/other/mult-pkg/faq/general/faq-how-do-i-interpret-odds-ratios-in-logistic-regression/
    return np.exp(x)


def logit_from_cohen_d(d): 
    # log odds ratio from Cohen's d
    # https://www.escal.site/
    # https://easystats.github.io/effectsize/reference/d_to_r.html
    return np.pi*d / math.sqrt(3)


def cohen_d_from_logit(x): 
    # Cohen's d from log odds ratio
    # https://stats.stackexchange.com/questions/68290/converting-odds-ratios-to-cohens-d-for-meta-analysis
    # https://www.um.es/metaanalysis/pdf/7078.pdf
    return x * math.sqrt(3) / np.pi


def logit_from_OR(x):
    return np.log(x)