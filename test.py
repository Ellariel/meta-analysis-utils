
import pandas as pd
import numpy as np
import statsmodels.api as sm
from metatools.report import *
from metatools.lm import lm

def s(x):
    return f'{x:.2f}'

test_data = sm.datasets.get_rdataset("mtcars", "datasets", cache=True).data

y = 'mpg'
x = ['wt', 'vs']
results, info = lm(test_data, y, x, model=['ols', 'rlm', 'glm'], 
                    verbose=True, 
                    constant=True,
                    standardized=False,
                    r_sq = True,
                    pred_r_sq = True,
                    ols_fit_cov_type='HC1', 
                    rlm_model_M=sm.robust.norms.RamsayE(),
                    glm_fit_cov_type='HC1',
                    glm_model_family=sm.families.Gamma())

print(info)

# OLS
assert s(results[0].fvalue) == '40.58'
assert s(results[0].params[1]) == '-4.44'
assert s(info[0]['pred_r_sq']) == '0.75'
# RLM
assert s(results[1].params[1]) == '-4.44'
assert s(info[1]['pred_r_sq']) == '0.75'
# GLM
assert s(results[2].pvalues[1]) == '0.00'
assert s(info[2]['pred_r_sq']) == '0.80'

