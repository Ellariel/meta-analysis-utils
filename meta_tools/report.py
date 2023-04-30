# !pip install factor-analyzer
# !pip install semopy

import math, random, scipy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import decomposition, preprocessing
from factor_analyzer import FactorAnalyzer
from factor_analyzer import (ConfirmatoryFactorAnalyzer, ModelSpecificationParser)

from scipy.stats import wilcoxon, mannwhitneyu
from scipy.stats import pearsonr, spearmanr

######################################################################################
# https://cran.r-project.org/web/packages/effectsize/vignettes/interpret.html
### FORMAT
def get_stars(p, p001='***', p01='**', p05='*', p10=''):
    if p < 0.001:
        return p001
    if p < 0.010:
        return p01
    if p < 0.050:
        return p05
    if p < 0.100:
        return p10
    return ''

def format_p(p):
    if p >= 0.999:
      raise 'bad p-value'
    if p < 0.0001:
        return 'p < .0001'
    if p < 0.001:
        return 'p < .001'
    return 'p = ' + f'{p:.3f}'[1:]

### STATISTICS
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

def t_from_b(b, se): # t-statistic from beta-coefficient
    return b/se

def t_from_z(z): # t-statistic from Fisher's z
    # https://web.cortland.edu/andersmd/STATS/stdscore.html#:~:text=As%20evidenced%20above%2C%20zscores%20are,a%20T%20score%20of%2025.
    return 10*z + 50

def z_from_p(p, method='two-tailed'): # Fisher's z from p-value
    # https://www.gigacalculator.com/calculators/p-value-to-z-score-calculator.php
    z = scipy.stats.norm.ppf(p/2) if method == 'two-tailed' else scipy.stats.norm.ppf(p)
    return -z

def cohen_r_from_p(p, n, method='two-tailed'): # Cohen’s r from p-value
    # Cohen’s guidelines for r are that a large effect is .5, a medium effect is .3, and a small effect is .1 (Coolican, 2009, p. 395).
    # Fritz, C. O., Morris, P. E., & Richler, J. J. (2012). Effect size estimates: Current use, calculations, and interpretation. Journal of Experimental Psychology: General, 141(1), 2–18. https://doi.org/10.1037/a0024338
    # https://www.researchgate.net/publication/51554230_Effect_Size_Estimates_Current_Use_Calculations_and_Interpretation
    return z_from_p(p, method=method) / math.sqrt(n)
    
def cohen_r_from_z(z, n): # Cohen’s r from Fisher's z 
    # Cohen’s guidelines for r are that a large effect is .5, a medium effect is .3, and a small effect is .1 (Coolican, 2009, p. 395).
    # Fritz, C. O., Morris, P. E., & Richler, J. J. (2012). Effect size estimates: Current use, calculations, and interpretation. Journal of Experimental Psychology: General, 141(1), 2–18. https://doi.org/10.1037/a0024338
    # https://www.researchgate.net/publication/51554230_Effect_Size_Estimates_Current_Use_Calculations_and_Interpretation
    return z / math.sqrt(n)

def cohen_d_from_z(z, n): # Cohen’s d from Fisher's z 
    # https://easystats.github.io/effectsize/reference/t_to_r.html
    return 2*z / math.sqrt(n)

def cohen_d_from_t(t, n): # Cohen’s d from t-statistic
    # Fritz, C. O., Morris, P. E., & Richler, J. J. (2012). Effect size estimates: Current use, calculations, and interpretation. Journal of Experimental Psychology: General, 141(1), 2–18. https://doi.org/10.1037/a0024338
    # https://www.researchgate.net/publication/51554230_Effect_Size_Estimates_Current_Use_Calculations_and_Interpretation
    return 2*t / math.sqrt(n-2)

def cohen_d_from_f(f, df_num, df_denom): # Cohen’s d from F-statistic
    # Cohen’s d derived from F-statistic (Fritz et al., 2012)
    # Fritz, C. O., Morris, P. E., & Richler, J. J. (2012). Effect size estimates: Current use, calculations, and interpretation. Journal of Experimental Psychology: General, 141(1), 2–18. https://doi.org/10.1037/a0024338
    # https://www.researchgate.net/publication/51554230_Effect_Size_Estimates_Current_Use_Calculations_and_Interpretation
    # https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00863/full
    eta_sq = (f * df_num) / (f * df_num + df_denom)
    d = 2 * math.sqrt(eta_sq) / math.sqrt(1 - eta_sq)
    return d

def z_from_r(r): # Fisher's z from correlation r
    # https://www.escal.site/
    return (np.log(1 + r) - np.log(1 - r)) / 2

def logOR_from_cohen_d(d): # log odds ratio from Cohen's d
    # https://www.escal.site/
    return np.pi*d / math.sqrt(3)

def r_from_cohen_d(d): # Pearson's r from Cohen's d
    return d / math.sqrt(d*d + 4)
   
### OLS   
def ols_APA(ols): # R² = .34, F(1, 416) = 6.71, p = .009
    r_sq, df_model, df_resid, fvalue = f'{ols.rsquared:.2f}'[1:], f'{ols.df_model:.0f}', f'{ols.df_resid:.0f}', f'{ols.fvalue:.2f}'
    return f"R² = {r_sq}, F({df_model}, {df_resid}) = {fvalue}, {format_p(ols.f_pvalue)}"

def calc_ols(data, x, y, standardized=True):
    cols = [y] + x
    d = data[cols]
    if standardized:
        d = preprocessing.scale(d)
        d = pd.DataFrame(d)
        d.columns = cols
    f = f'{y} ~ 1 + ' + '+'.join(x)
    print(f)
    _lm = smf.ols(f, d).fit(cov_type='HC1')
    rlm = smf.rlm(f, d, M=sm.robust.norms.HuberT()).fit() #TrimmedMean(0.5)

    base_ols = pd.read_html(_lm.summary().tables[1].as_html(), header=0, index_col=0)[0].rename(columns={'P>|z|' : 'p-value',
                                                                                  'std err' : 'std_err',
                                                                                  '[0.025' : 'CIL',
                                                                                  '0.975]' : 'CIR'})
    base_ols['sig'] = [get_stars(i) for i in base_ols['p-value']]
    robust_ols = pd.read_html(rlm.summary().tables[1].as_html(), header=0, index_col=0)[0].rename(columns={'P>|z|' : 'p-value',
                                                                                  'std err' : 'std_err',
                                                                                  '[0.025' : 'CIL',
                                                                                  '0.975]' : 'CIR'})
    robust_ols['sig'] = [get_stars(i) for i in robust_ols['p-value']]
    robust_ols.columns = [i+'_robust' for i in robust_ols.columns]
    f = pd.Series(x).rename('index')
    f = pd.concat([f, f.rename('i')], axis=1).set_index('i')
    robust_ols = f.join(robust_ols, how='left')
    result = robust_ols.join(base_ols, how='left')
    apa = ols_APA(_lm)
    print(apa)
    return result.reset_index(drop=True), apa, _lm, rlm

### FACTOR ANALYSIS
def factor_analysis(data, n_factors=1, rotation=None, sort=False, rename_dict=None):
    d, cols = data.dropna(), data.columns.to_list()
    d = preprocessing.scale(d)
    fa = FactorAnalyzer(rotation=rotation, n_factors=1)
    fa.fit(d)
    t0 = pd.concat([pd.Series([i if not rename_dict else rename_dict[i] for i in cols]),
                    pd.DataFrame(fa.loadings_)], axis=1)
    t0.columns = ['index', 'f0']
    t0.set_index('index', inplace=True)
    t0 = t0.apply(lambda x: x.apply(lambda y: f'{y:.2f}'))
    #
    fa = FactorAnalyzer(rotation=None, n_factors=n_factors)
    fa.fit(d)
    t1 = pd.concat([pd.Series([i if not rename_dict else rename_dict[i] for i in cols]),
                    pd.DataFrame(fa.loadings_)], axis=1)
    t1.columns = ['index'] + [f'f{i+1}' for i in range(fa.n_factors)]
    t1.set_index('index', inplace=True)
    t1 = t1.apply(lambda x: x.apply(lambda y: f'{y:.2f}'))
    #
    f = pd.concat([t0, t1], ignore_index=False, axis=1)
    if sort:
        f.sort_values(by=sort, ascending=False, inplace=True)
    return f

### PLOTS
def errorbars(data, method='std'):
    if method == 'std':
        y = np.std(data)
    return y

def plot_ecdf(data, xlabel=None, ylabel=None):
    fig = plt.figure()
    ax = fig.add_subplot()
    sns.ecdfplot(data = data, legend = data)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel("ECDF")
    else:
        plt.ylabel(ylabel)
    #plt.show()
