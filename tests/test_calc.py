
import statsmodels.api as sm
from metatools.calc import *

def s(x):
    return f'{x:.3f}'

def test_calc():
    ##############
    # conversion #
    ##############
    
    r = 0.75
    d = d_from_r(r)
    assert s(d) == '2.268'
    r = r_from_d(d)
    assert s(r) == '0.750'
    
    r = 0.5
    z = z_from_r(r)
    r = r_from_z(z)
    assert s(z) == '0.549'
    assert s(r) == '0.500'
    
    r = 0.9
    n = 100
    t = t_from_r(r, n)
    z = z_from_r(r)
    assert s(t) == '20.440'
    assert s(z) == '1.472'
    r = r_from_t(t, n)
    assert s(r) == '0.900'
    r = r_from_z(z)
    assert s(r) == '0.900'   
    d = d_from_t(t, n)
    assert s(d) == '4.129'
    r = r_from_d(d)
    assert s(r) == '0.900'
    d = d_from_z(z)
    assert s(d) == '4.129'
    r = r_from_d(d)
    assert s(r) == '0.900'
    z = z_from_d(d)
    assert s(z) == '1.472'    
    t = t_from_d(d, n)
    assert s(t) == '20.440'
    
  
    ###########
    # p-value #
    ###########  

    r = -0.181
    n = 181
    p = p_from_r(r, n, method='two-tailed')
    assert s(p) == '0.015'
    r = r_from_p(p, n, method='two-tailed')
    assert s(r) == '0.181'  # does not keep sign 
    
    d = -0.41
    n = 181
    p = p_from_d(d, n, method='two-tailed') 
    assert s(p) == '0.007'
    d = d_from_p(p, n, method='two-tailed')
    assert s(d) == '0.410' # does not keep sign   
    
    t = -2.16
    n = 163
    p = p_from_t(t, n, method='two-tailed')
    assert s(p) == '0.032'
    t = t_from_p(p, n, method='two-tailed')
    assert s(t) == '2.160'
    t = 2.16
    p = p_from_t(t, n, method='two-tailed')
    assert s(p) == '0.032'
    t = t_from_p(p, n, method='two-tailed')
    assert s(t) == '2.160'
    
    p = 0.5
    z = z_from_p(p, method='two-tailed')
    p = p_from_z(z, method='two-tailed')
    assert s(z) == '0.674'
    assert s(p) == '0.500'    
    z = -2.335
    p = p_from_z(z, method='two-tailed')
    assert s(p) == '0.020'
    z = 2.052
    p = p_from_z(z, method='two-tailed')
    assert s(p) == '0.040'
    z = z_from_p(p, method='two-tailed')
    assert s(z) == '2.052'
    
    n = 163
    z = -2.335
    p = p_from_z(z, method='two-tailed')
    t = t_from_p(p, n, method='two-tailed')
    p = p_from_t(t, n, method='two-tailed')
    assert s(p) == '0.020'
    z = z_from_p(p)
    assert s(z) == '2.335'
    
    t = -2.16
    p = p_from_t(t, n, method='two-tailed')
    z = z_from_p(p, method='two-tailed')
    p = p_from_z(z, method='two-tailed')
    assert s(p) == '0.032'
    t = t_from_p(p, n, method='two-tailed')
    assert s(t) == '2.160'   
    
    d = 2.5
    n = 10
    t = t_from_d(d, n)
    assert s(t) == '3.536'
    d = d_from_t(t, n)
    assert s(d) == '2.500'  
    z = z_from_t(t, n)
    d = d_from_z(z)
    assert s(d) == '2.500' 
    
    d = 2.5
    n = 10
    z = z_from_d(d)
    assert s(z) == '1.048'
    d = d_from_z(z)
    assert s(d) == '2.500'  
    t = t_from_z(z, n)
    d = d_from_t(t, n)
    assert s(d) == '2.500'   
    
    r = 0.9
    n = 10
    t = t_from_r(r, n)
    assert s(t) == '5.840'
    r = r_from_t(t, n)
    assert s(r) == '0.900'  
    z = z_from_t(t, n)
    r = r_from_z(z)
    assert s(r) == '0.900'   
    
    r = 0.90
    n = 10
    z = z_from_r(r)
    assert s(z) == '1.472'
    r = r_from_z(z)
    assert s(r) == '0.900'  
    t = t_from_z(z, n)
    r = r_from_t(t, n)
    assert s(r) == '0.900'  
 
    p = 0.010
    n = 10
    t = t_from_p(p, n, method='two-tailed')
    assert s(t) == '3.355'
    p = p_from_t(t, n, method='two-tailed')
    assert s(p) == '0.010'   
    z = z_from_t(t, n, method='two-tailed')
    p = p_from_z(z, method='two-tailed')
    assert s(p) == '0.010' 
 
    p = 0.010
    n = 10
    z = z_from_p(p, method='two-tailed')
    assert s(z) == '2.576'
    p = p_from_z(z, method='two-tailed')
    assert s(p) == '0.010'  
    t = t_from_z(z, n, method='two-tailed')
    p = p_from_t(t, n)
    assert s(p) == '0.010' 
 
    # sign check
    
    r = -0.9
    n = 10
    t = t_from_r(r, n)
    assert s(t) == '-5.840'
    r = r_from_t(t, n)
    assert s(r) == '-0.900'  
    z = z_from_t(t, n)
    assert s(z) == '-1.472'
    r = r_from_z(z)
    assert s(r) == '-0.900' 
    
    r = -0.90
    n = 10
    z = z_from_r(r)
    assert s(z) == '-1.472'
    r = r_from_z(z)
    assert s(r) == '-0.900'  
    t = t_from_z(z, n)
    assert s(t) == '-5.840'
    r = r_from_t(t, n)
    assert s(r) == '-0.900'
    
    d = -2.5
    n = 10
    t = t_from_d(d, n)
    assert s(t) == '-3.536'
    d = d_from_t(t, n)
    assert s(d) == '-2.500'  
    z = z_from_t(t, n)
    assert s(z) == '-1.048'
    d = d_from_z(z)
    assert s(d) == '-2.500' 
    
    d = -2.5
    n = 10
    z = z_from_d(d)
    assert s(z) == '-1.048'
    d = d_from_z(z)
    assert s(d) == '-2.500'  
    t = t_from_z(z, n)
    assert s(t) == '-3.536'
    d = d_from_t(t, n)
    assert s(d) == '-2.500' 

    
    
    #print(r)
     
    '''
    
    
    r = 0.5
    n = 20
    z = z_from_r(r)
    assert s(z) == '0.549'
    t = t_from_z(z, n)
    print('t_from_z', t)
    d = d_from_t(t, n)
    print('d_from_t', d)
    r = r_from_d(d)
    print(r)
    assert s(r) == '0.500'    
    
    
    
    print('\n')

    t = t_from_z(z, n)
    p = p_from_t(t, n, method='two-tailed')
    print('p_from_t', p)
    print('t_from_z', t)
    
    t = -2.16
      
    z = z_from_t(t, n)
    p = p_from_z(z, method='two-tailed')
    print('z_from_t', z)
    print('p_from_z', p)
    '''

    '''    
    test_data = sm.datasets.get_rdataset("mtcars", "datasets", cache=True).data
    assert len(test_data) == 32

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
    # OLS
    assert s(results[0].fvalue) == '40.58'
    assert s(results[0].params.iloc[1]) == '-4.44'
    assert s(info[0]['pred_r_sq']) == '0.75'
    # RLM
    assert s(results[1].params.iloc[1]) == '-4.44'
    assert s(info[1]['pred_r_sq']) == '0.75'
    # GLM
    assert s(results[2].pvalues.iloc[1]) == '0.00'
    assert s(info[2]['pred_r_sq']) == '0.80'

    results_APA = lm_APA(results, info)
    print(results_APA)

    # lm_APA OLS
    assert results_APA[0].loc['vs']['p-value'] == '.002'
    assert results_APA[0].iloc[0]['model'] == 'R² = .80, R²adj = .79, R²pred = .75, F(2, 30) = 40.58, p < .001'
    # lm_APA RLM
    assert results_APA[1].loc['vs']['p-value'] == '.017'
    assert results_APA[1].iloc[0]['model'] == 'R² = .89, R²adj = .89, R²pred = .75, F(2, 30) = 126.93, p < .001'
    # lm_APA GLM
    assert results_APA[2].loc['vs']['p-value'] == '.003'
    assert results_APA[2].iloc[0]['model'] == 'R² = .99, R²adj = .99, R²pred = .80, F(2, 30) = 1520.63, p < .001'

    print('Tests for metatools.lm are PASSED!')
    '''
    
if __name__ == '__main__':
    test_calc()
