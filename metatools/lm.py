import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm
from sklearn.model_selection import LeaveOneOut
from .diagnostic import vif


def fit_model(model, Y, X, **kwargs): # model.fit() generator function
    if isinstance(model, (list, tuple)):
        for m in model:
            yield next(fit_model(m, Y, X, **kwargs))
    elif isinstance(model, str):
        if model == 'ols':
            yield next(fit_model(sm.OLS, Y, X, **kwargs))
        elif model == 'rlm':
            yield next(fit_model(sm.RLM, Y, X, **kwargs))
        elif model == 'glm':
            yield next(fit_model(sm.GLM, Y, X, **kwargs))
        else:
            raise NotImplementedError(f"{model} is not implemented, try 'ols', 'rlm' or 'glm'.")
    else:
        verbose = kwargs.get('verbose', False)
        if model == sm.OLS:
            if verbose:
                print('model: OLS')
            _kwargs = {k[4:] : v for k, v in kwargs.items() if k.startswith('ols_')}
        elif model == sm.RLM:
            if verbose:
                print('model: RLM')
            _kwargs = {k[4:] : v for k, v in kwargs.items() if k.startswith('rlm_')}
        elif model == sm.GLM:
            if verbose:
                print('model: GLM')
            _kwargs = {k[4:] : v for k, v in kwargs.items() if k.startswith('glm_')}
        else:
            _kwargs = kwargs
        model_kwargs = {k[6:] : v for k, v in _kwargs.items() if k.startswith('model_')}
        fit_kwargs = {k[4:] : v for k, v in _kwargs.items() if k.startswith('fit_')}

        if verbose:
            print('model_kwargs:', model_kwargs)
            print('fit_kwargs:', fit_kwargs)

        yield model(Y, X, **model_kwargs).fit(**fit_kwargs)


def pred_r_sq(model, Y, X, **kwargs):
    '''
    Calculating R²pred for statsmodels

    https://stats.stackexchange.com/questions/592653/how-to-get-predicted-r-square-from-statmodels
    '''
    res = {}
    errors = []
    kwargs['verbose'] = False
    for train_index, test_index in LeaveOneOut().split(X):
                x_train, x_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
                try:
                    for idx, r in enumerate(fit_model(model, y_train, x_train, **kwargs)):
                        res.setdefault(idx, [])
                        res[idx].append(*(y_test - r.predict(x_test)))
                except Exception as e:
                        errors.append(str(e))
    if len(errors):
        print(f"Some attempts to calculate R²pred have been unsuccessful ({len(errors)}): {set(errors)}")
    return np.clip([1 - np.sum(np.square(i)) / np.var(Y) / Y.size for _, i in res.items()], -1.0, 1.0)


def r_sq(results):
    '''
    Calculating/retrieving R²/R²pseudo, R²adj, R²pred
    for OLS, RLM, GLM from statsmodels fitting results

    https://stats.stackexchange.com/questions/83826/is-a-weighted-r2-in-robust-linear-model-meaningful-for-goodness-of-fit-analys
    https://stats.stackexchange.com/questions/55236/prove-f-test-is-equal-to-t-test-squared
    '''

    weights = getattr(results, 'weights', 1)
    resid = getattr(results, 'resid', 
                getattr(results, 'resid_working', None))
    nobs = int(results.nobs)
    n_pred = len(set(results.params.index) - set(['const'])) # number of predictors, without intercept
    df_model = max(int(results.df_model), n_pred) # technical correction
    df_resid = max(int(results.df_resid), nobs - df_model) # technical correction
    fitted = results.fittedvalues
    observed = resid + fitted    
    SSe = np.sum((weights * resid) ** 2)
    SSt = np.sum((weights * (observed - np.mean(weights * observed)) ) ** 2)
    r_sq = getattr(results, 'rsquared', 
                getattr(results, 'pseudo_rsquared', 1 - SSe/SSt))
    r_sq = r_sq() if callable(r_sq) else r_sq
    
    # https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLSResults.rsquared_adj.html
    r_sq_adj = getattr(results, 'rsquared_adj', 1 - (1 - r_sq) * nobs / df_resid)

    # https://www.slideshare.net/slideshow/multiple-regressionppt-252604177/252604177#8
    f_stat = getattr(results, 'fvalue', (r_sq / df_model) / ((1 - r_sq) / df_resid)) # (SSt / df_model) / (SSe / df_resid)
    f_pvalue = getattr(results, 'f_pvalue', scipy.stats.f.sf(f_stat, df_model, df_resid))

    return {'r_sq': r_sq,
            'r_sq_adj': r_sq_adj,
            'df_model': df_model,
            'df_resid': df_resid,
            'f_stat': f_stat,
            'f_pvalue': f_pvalue}
      

def lm(data, y, x, model='ols', **kwargs):
    '''
    Fitting OLS, RLM, GLM from statsmodels

    lm(test_data, Y, X, model=['ols', 'rlm'], 
                    verbose=True,
                    constant=True,
                    standardized=False, # keeps np.number columns only
                    r_sq = True, 
                    pred_r_sq = True,
                    ols_fit_cov_type='HC1', 
                    rlm_model_M=sm.robust.norms.RamsayE())
    '''

    verbose = kwargs.get('verbose', True)
    constant = kwargs.pop('constant', True)
    standardized = kwargs.pop('standardized', False)
    add_r_sq = kwargs.pop('r_sq', False)
    add_pred_r_sq = kwargs.pop('pred_r_sq', False)
    calc_vif = kwargs.pop('vif', False)

    if verbose and constant and standardized:
        print('Having constant=True and standardized=True at the same time does not make sense and can lead to errors.')

    df = data[[y] + x].dropna()
    if verbose and len(df) != len(data):
        print(f"N={len(data)}")
        print('Rows with NAs were dropped!')

    if standardized:
        df = df.select_dtypes(include=[np.number, 'bool']).apply(scipy.stats.zscore)
    X, Y = df[x], df[y]

    if verbose:
        print(f"N={len(Y)}")
        print(f"formula: {y} ~ {'1 + ' if constant else ''}" + " + ".join(x))

    if constant:
        X = sm.add_constant(X)

    results = []
    for r in fit_model(model, Y, X, **kwargs):
        results.append(r)
        if verbose:
            print(r.summary())

    info = []
    if add_r_sq or add_pred_r_sq:
        info = [r_sq(r) for r in results]
        if add_pred_r_sq:
            info = [{**r, **{'pred_r_sq': rr}} 
                        for r, rr in zip(info, pred_r_sq(model, Y, X, **kwargs))]
    if calc_vif:
        info = [{**i, **{'vif': vif(r)}} 
                    for i, r in zip(info, results)]

    return results, info
