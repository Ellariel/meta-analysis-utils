import numpy as np


###############
#  formating  #
###############


def get_stars(p, p001="***", p01="**", p05="*", p10="⁺", p_=""):
    if p < 0.001:
        return p001
    if p < 0.010:
        return p01
    if p < 0.050:
        return p05
    if p < 0.100:
        return p10
    return p_


def format_p(p, keep_spaces=True, keep_p=True):
    if not np.isfinite(p) or p < 0.0:
        p = "= inf"
    elif p < 0.0001:
        p = "< .0001"
    elif p < 0.001:
        p = "< .001"
    elif p > 0.99:
        p = "= 1."
    else:
        p = "= " + f"{p:.3f}"[1:]
    if keep_p:
        p = "p " + p
    if not keep_spaces:
        p = p.replace(" ", "")
    return p


def format_r(r, keep_spaces=True, keep_r=True):
    if not np.isfinite(r):
        r = "= inf"
    elif np.abs(r) < 0.01:
        r = "= 0."
    elif np.abs(r) > 0.99:
        r = "= 1."
    else:
        r = "= " + f"{'-' if r < 0.0 else ''}" + f"{np.abs(r):.2f}"[1:]
    if keep_r:
        r = "r " + r
    if not keep_spaces:
        r = r.replace(" ", "")
    return r

