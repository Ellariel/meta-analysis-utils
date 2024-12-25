# A Toolbox for Meta-Analysis and APA-Compliant Reporting

## About
This toolbox is a curated collection of utilities I frequently use for statistical analysis and nicely reporting research findings in accordance with APA standards. While some of the included libraries may be slightly outdated, they remain functional and reliable, making them my go-to tools.

## Install
```shell
pip install git+https://github.com/ellariel/metatools.git
```

## Content and examples
* ***metatools.calc*** - *a basic module for statistical conversions and bootstrapping*
* ***metatools.lm*** - *a module offering wrapper functions for statsmodels, streamlining the fitting of OLS, RLM, and GLM models, as well as generating detailed tables*
* ***metatools.sem*** - *a module for performing structural equation modeling, complete with result reporting and path diagram visualization*
* ***metatools.format*** - *simple APA-compliant formatting functions for numerical results*

### *metatools.calc*

This module provides essential conversion functions alongside tools for bootstrapping means, confidence intervals, and related statistics. It includes a comprehensive set of approximate conversion functions for bidirectional transformations between t-, z-, and F-statistics, as well as conversions involving Cohen's r, Cohen's d, and p-values.

#### Conversion examples
```python
from metatools.calc import cohen_r_from_z, cohen_d_from_z
# calculation Cohen's r and d from z-score:
# there is also a shorten aliases r_from_z and d_from_z
z = -1.472
r = cohen_r_from_z(z) # r = -0.9
d = cohen_d_from_z(z) # d = -4.1
```

```python
from metatools.calc import p_from_r
# calculation p-value for Cohen's r  or correlation coefficient:
r = -0.79
n = 100
p = p_from_r(r, n) # p = 0.0
```

#### Bootstrap examples
```python
from metatools.calc import bootstrap
# bootstraping confidence intervals for a mean value:
cil, m, cir = bootstrap([1, 2, 3, 4, 5, 6, 7, 8, 9],
        func=np.mean,
        n_rep=1000) # cil = 3.3, m = 5.0, cir = 6.7
```

```python
from metatools.calc import bootstrap
from scipy.stats import spearmanr
# bootstraping value for Spearman's correlation:
def func(*args):
    return spearmanr(*args).statistic

_, m, _ = bootstrap(
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    [4, 5, 8, 4, 5, 6, 7, 8, 9],
    func=func,
    n_rep=1000) # m = 0.75
```


### *metatools.format*
Simple APA-compliant functions for formatting numeric results, such as formatting R and r values, p-values, etc.

#### Formatting examples
```python
from metatools.format import get_stars, format_p, format_r
# formatting numeric results
r = format_r(0.999, keep_spaces=True, use_letter="r", no_equals=False) # r = 'r = 1.'
r = format_r(0.011, keep_spaces=True, use_letter="R", no_equals=False) # r = 'R = .01'
p = format_p(0.0004, use_letter="p", keep_spaces=True, no_equals=False) # p = 'p < .001'
s = get_stars(0.04, p001="***", p01="**", p05="*", p10="⁺", p_="") # s = '*'
```


## Sources and references
* [factor-analyzer](https://pypi.org/project/factor-analyzer/) (GPL-2.0 License)
* [statsmodels](https://www.statsmodels.org/stable/) (BSD License)
* [semopy](https://pypi.org/project/semopy/) (MIT License)
* [scipy](https://pypi.org/project/scipy/) (BSD License)

