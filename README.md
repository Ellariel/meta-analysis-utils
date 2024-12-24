# A Toolbox for Meta-Analysis and APA-Compliant Reporting

## About
This toolbox is a curated collection of utilities I frequently use for statistical analysis and nicely reporting research findings in accordance with APA standards. While some of the included libraries may be slightly outdated, they remain functional and reliable, making them my go-to tools.

## Install
```shell
pip install git+https://github.com/ellariel/metatools.git
```

## Examples
* ***metatools.calc*** - *Basic Module for Statistical Conversions and Bootstrapping*
* metatools.format
* metatools.lm
* metatools.sem

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
def f(*args):
    return spearmanr(*args).statistic

_, m, _ = bootstrap(
    [1, 2, 3, 4, 5, 6, 7, 8, 9],
    [4, 5, 8, 4, 5, 6, 7, 8, 9],
    func=f,
    n_rep=1000) # m = 0.75
```



