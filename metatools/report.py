# !pip install factor-analyzer
# !pip install semopy
# !pip install lxml

import math, random, scipy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn import decomposition, preprocessing
from itertools import zip_longest

from factor_analyzer import FactorAnalyzer
from factor_analyzer import (ConfirmatoryFactorAnalyzer, ModelSpecificationParser)
import semopy as sem

#from scipy.stats import wilcoxon, mannwhitneyu
#from scipy.stats import pearsonr, spearmanr

import warnings
warnings.filterwarnings("ignore")

######################################################################################
# https://cran.r-project.org/web/packages/effectsize/vignettes/interpret.html
### FORMAT


def get_stars(p, p001='***', p01='**', p05='*', p10='⁺'):
    if p < 0.001:
        return p001
    if p < 0.010:
        return p01
    if p < 0.050:
        return p05
    if p < 0.100:
        return p10
    return ''


def format_p(p, add_p=True, keep_space=True):
    if p >= 0.999:
        p = 'p = 1.000'
    elif p < 0.001:
        p = 'p < .001'
    elif p < 0.0001:
        p = 'p < .0001'
    elif not np.isfinite(p) or p < 0.0:
        p = 'p = inf'
    else:
        p = 'p = ' + f"{p:.3f}"[1:]
    p = p if add_p else p.replace('p ', '').replace('=', '')
    p = p if keep_space else p.replace(' ', '')
    return p


def format_r(r):
    if abs(r) >= 0.999:
        r = '1.00'
    elif abs(r) < 0.005:
        r = '0.00'
    elif not np.isfinite(r):
        r = 'inf'
    else:
        r = f"{'-' if r < 0.0 else ''}"+f"{abs(r):.2f}"[1:]
    return r


def lm_APA(results, info={}, decimal=None): 
    # R² = .34, R²adj = .34, R²pred = .34, F(1, 416) = 6.71, p = .009
    res = []
    for r, i in zip_longest(results, info, fillvalue={}):
        if 'pred_r_sq' not in i:
            if 'r_sq' in i:
                i = f"R² = {format_r(i['r_sq'])}, R²adj = {format_r(i['r_sq_adj'])}, F({i['df_model']}, {i['df_resid']}) = {i['f_stat']:.2f}, {format_p(i['f_pvalue'])}"
            else:
                i = ''
        else:
            i = f"R² = {format_r(i['r_sq'])}, R²adj = {format_r(i['r_sq_adj'])}, R²pred = {format_r(i['pred_r_sq'])}, F({i['df_model']}, {i['df_resid']}) = {i['f_stat']:.2f}, {format_p(i['f_pvalue'])}"
        params = pd.read_html(r.summary().tables[1].as_html(), header=0, index_col=0)[0]\
                                .rename(columns={'P>|z|' : 'p-value',
                                                 'std err' : 'std_err',
                                                 '[0.025' : 'CIL',
                                                 '0.975]' : 'CIR'})
        if decimal:
            for c in ['coef', 'std_err', 'CIL', 'CIR']:
                params[c] = params[c].apply(lambda x: round(x, decimal))
        params['sig'] = [get_stars(c) for c in params['p-value']]
        params['p-value'] = [format_p(c, add_p=False, keep_space=False) for c in params['p-value']]
        if len(i):
            params['model'] = ''
            params['model'].iloc[0] = i
        res.append(params)
    return res


#for r, i in zip_longest(results, info, fillvalue={}):
### STATISTICS
def CI_for_mean(data, func=np.mean, p=0.95, n=1000, seed=13):
    # Bootstrapped 95% confidence intervals for the mean/median value from 1000 resamples are reported.
    # https://towardsdatascience.com/how-to-calculate-confidence-intervals-in-python-a8625a48e62b
    random.seed(seed)
    np.random.seed(seed)
    simulations = [func(np.random.choice(data, size=len(data), replace=True)) for i in range(n)]
    lp, rp, m = (1-p)/2, 1-(1-p)/2, func(data)
    return np.hstack([m, np.percentile(simulations, [lp*100, rp*100])])

def CI_for_r(r, n, p=0.95):
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

def cohen_d_from_r(r): # Cohen's d from correlation r  
    # https://easystats.github.io/effectsize/reference/d_to_r.html
    return math.sqrt(4) * r / math.sqrt(1 - r*r) # equal groups

def z_from_r(r): # Fisher's z from correlation r
    # https://www.escal.site/
    # https://en.wikipedia.org/wiki/Fisher_transformation
    return math.atanh(r) # np.log((1 + r)/(1 - r)) / 2

def r_from_cohen_d(d): # Сorrelation r from Cohen's d
    # https://www.escal.site/
    # https://easystats.github.io/effectsize/reference/d_to_r.html
    return d / math.sqrt(d*d + 4) # equal groups

def r_from_f(f, df): # Сorrelation r from F-statistic
    # https://juls-dotcom.github.io/meta_analysis.html
    return f / math.sqrt(f + df)
    
def r_from_t(t, df): # Сorrelation r from t-statistic
    # https://juls-dotcom.github.io/meta_analysis.html
    return t*t / math.sqrt(t*t + df)

def r_from_z(z, n): # Сorrelation r (Cohen’s r) from Fisher's z 
    # https://juls-dotcom.github.io/meta_analysis.html
    return z / math.sqrt(n)

def r_from_p(p, n, method='two-tailed'): # Сorrelation r (Cohen’s r) from p-value
    return cohen_r_from_p(p, n, method=method)

def unbiased_z(z): # unbiased Z
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

def OR_from_logit(x): # odds ratio from log odds ratio
    # odds = EXP(0.873) = 2.394;
    # https://stats.oarc.ucla.edu/other/mult-pkg/faq/general/faq-how-do-i-interpret-odds-ratios-in-logistic-regression/
    return np.exp(x)

def logit_from_cohen_d(d): # log odds ratio from Cohen's d
    # https://www.escal.site/
    # https://easystats.github.io/effectsize/reference/d_to_r.html
    return np.pi*d / math.sqrt(3)

def cohen_d_from_logit(x): # Cohen's d from log odds ratio
    # https://stats.stackexchange.com/questions/68290/converting-odds-ratios-to-cohens-d-for-meta-analysis
    # https://www.um.es/metaanalysis/pdf/7078.pdf
    return x * math.sqrt(3) / np.pi

def logit_from_OR(x):
    return np.log(x)
   
### OLS   
def ols_APA(ols): # R² = .34, F(1, 416) = 6.71, p = .009
    r_sq, df_model, df_resid, fvalue = f'{ols.rsquared:.2f}'[1:], f'{ols.df_model:.0f}', f'{ols.df_resid:.0f}', f'{ols.fvalue:.2f}'
    return f"R² = {r_sq}, F({df_model}, {df_resid}) = {fvalue}, {format_p(ols.f_pvalue)}"
    
def rlm_APA(rlm): # R² = .34, F(1, 416) = 6.71, p = .009
    # https://stats.stackexchange.com/questions/55236/prove-f-test-is-equal-to-t-test-squared
    # https://stats.stackexchange.com/questions/83826/is-a-weighted-r2-in-robust-linear-model-meaningful-for-goodness-of-fit-analys?answertab=modifieddesc#tab-top
    SSe = np.sum((rlm.weights * rlm.resid) ** 2)
    observed = rlm.resid + rlm.fittedvalues
    SSt = np.sum((rlm.weights * (observed - np.mean(rlm.weights * observed)) ) ** 2)
    r_sq = 1-SSe/SSt
    f_stat = (SSt/rlm.df_model) / (SSe/rlm.df_resid)
    f_pvalue = scipy.stats.f.sf(f_stat, rlm.df_model, rlm.df_resid)
    r_sq, df_model, df_resid, fvalue = f'{r_sq:.2f}'[1:], f'{rlm.df_model:.0f}', f'{rlm.df_resid:.0f}', f'{f_stat:.2f}'
    return f"R² = {r_sq}, F({df_model}, {df_resid}) = {fvalue}, {format_p(f_pvalue)}"

def calc_ols(data, x, y, standardized=True, drop_rlm=False, ols_cov_type='HC1', rlm_cov_type='Huber'):
    cols = [y] + x
    d = data[cols]
    if standardized:
        d = preprocessing.scale(d)
        d = pd.DataFrame(d)
        d.columns = cols
    f = f'{y} ~ 1 + ' + ' + '.join(x)
    print(f)
    _lm = smf.ols(f, d).fit(cov_type=ols_cov_type)
    if rlm_cov_type=='Huber':
        rlm_cov_type = sm.robust.norms.HuberT()
    rlm = smf.rlm(f, d, M=rlm_cov_type).fit() #TrimmedMean(0.5)

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
    result = f.join(base_ols, how='left')
    ols_apa = ols_APA(_lm)
    rlm_apa = rlm_APA(rlm)
    print('OLS:' + ols_apa)    
    if not drop_rlm:
        print('ROLS:' + rlm_apa)
        result = result.join(robust_ols, how='left')

    return result.reset_index(drop=True), ols_apa, _lm, rlm_apa, rlm

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


### SEM

# https://arxiv.org/pdf/2106.01140.pdf
# formula = f"""
# ПП =~ {' + '.join(p)}
# НЭ =~ {' + '.join(e)}
# ПП ~ НЭ + {' + '.join(x)} + Группа
# """

def run_sem(data, formula, obj='MLW', solver='SLSQP', bootstrap=False, plot_covs=True, standardized=True, save_to_file='sem.pdf', seed=48):
    random.seed(seed)
    np.random.seed(seed)
    model = sem.Model(formula)
    model.fit(data, obj=obj, solver=solver) #MLW ULS GLS FIML DWLS WLS
    if bootstrap:
        sem.bias_correction(model, n=bootstrap, resample_mean=True)
    g = sem.semplot(model, save_to_file, plot_covs=plot_covs, std_ests=standardized, show=True)
    return g, sem.calc_stats(model).T, model.inspect()
