from metatools.calc import *
# calculation Cohen's r and d from z-score:
z = -1.472
r = cohen_r_from_z(z) # r = -0.9
d = cohen_d_from_z(z) # d = -4.1
r = -0.79
n = 100
p = p_from_r(r, n)
print(p)