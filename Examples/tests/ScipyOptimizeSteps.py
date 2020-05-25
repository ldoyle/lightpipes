# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:11:59 2020

https://scipy-cookbook.readthedocs.io/items/discrete_bvp.html

@author: Lenny
"""

import numpy as _np
from scipy.optimize import least_squares #TODO hide from public
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt

N = 50
c = 1
def f(u, ):
    return u**3

def f_prime(u):
    return 3 * u**2

def fun(u, n, f, f_prime, c, **kwargs):
    v = _np.zeros((n + 2, n + 2))
    u = u.reshape((n, n))
    v[1:-1, 1:-1] = u
    y = v[:-2, 1:-1] + v[2:, 1:-1] + v[1:-1, :-2] + v[1:-1, 2:] - 4 * u + c * f(u)
    return y.ravel()

def compute_jac_indices(n):
    i = _np.arange(n)
    jj, ii = _np.meshgrid(i, i)

    ii = ii.ravel() #counts up each row, primary index
    jj = jj.ravel() #counts up each col, sec. index, faster varying

    ij = _np.arange(n**2) #0....N**2-1, faster varying along row
    # 0 1 2 3
    # 4 5 5 6 ...

    jac_rows = [ij] #each value contributes to itself for sure
    jac_cols = [ij]

    mask = ii > 0 #all rows >= 1
    ij_mask = ij[mask]
    jac_rows.append(ij_mask) #add all "+1"
    jac_cols.append(ij_mask - n) #add all "-1"

    mask = ii < n - 1
    ij_mask = ij[mask]
    jac_rows.append(ij_mask)
    jac_cols.append(ij_mask + n)

    mask = jj > 0
    ij_mask = ij[mask]
    jac_rows.append(ij_mask)
    jac_cols.append(ij_mask - 1)

    mask = jj < n - 1
    ij_mask = ij[mask]
    jac_rows.append(ij_mask)
    jac_cols.append(ij_mask + 1)

    return _np.hstack(jac_rows), _np.hstack(jac_cols)
jac_rows, jac_cols = compute_jac_indices(N)
u0 = _np.ones(N**2) * 0.5
# u0 = Fin.field.ravel() #initial guess is old field

def jac(u, n, f, f_prime, c, jac_rows=None, jac_cols=None):
    jac_values = _np.ones_like(jac_cols, dtype=float)
    jac_values[:n**2] = -4 + c * f_prime(u)
    return coo_matrix((jac_values, (jac_rows, jac_cols)),
                      shape=(n**2, n**2))

joo = jac(u0.real, N, f, f_prime,c, jac_rows, jac_cols)
# res_2 = least_squares(fun, u0.real, jac=jac, gtol=1e-3,
#                       args=(N, f, f_prime, c),
#                       kwargs={'jac_rows': jac_rows,
#                               'jac_cols': jac_cols},
#                       verbose=0)

res_1 = least_squares(fun, u0.real, gtol=1e-3,
                      args=(N, f, f_prime, c),
                      kwargs={'jac_rows': jac_rows,
                              'jac_cols': jac_cols},
                      verbose=0)

# print(res_1)
uf = res_1.x.reshape((N, N))
plt.imshow(uf)