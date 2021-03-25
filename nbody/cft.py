#!/usr/bin/python
# -*- coding:utf-8 -*-

"""2D constrained field code example
Constrained Gaussian random field are easy! This tiny python program
computes a random field following some constraints; and it is fast!"""

import sys, time, struct
from math import pi

import numpy as np
from numpy import fft, random
from scipy.stats import chi2 as _chi2
from functools import reduce, partial
from numbers import Number
import operator

def _wave_number(s):
    N = s[0]
    i = np.indices(s)
    return np.where(i > N/2, i - N, i)

class Box:
    def __init__(self, dim, N, L):
        self.N = N
        self.L = L
        self.res = L/N
        self.dim = dim

        self.shape = (self.N,) * dim
        self.size = reduce(lambda x, y: x*y, self.shape)

        self.K = _wave_number(self.shape) * 2*pi / self.L
        self.k = np.sqrt((self.K**2).sum(axis=0))

        self.k_max = N*pi/L
        self.k_min = 2*pi/L

class Filter:
    def __init__(self, f):
        self.f = f

    def __call__(self, K):
        return self.f(K)

    def __mul__(self, other):
        if isinstance(other, Filter):
            return Filter(lambda K: self.f(K) * other.f(K))

        elif isinstance(other, Number):
            return Filter(lambda K: other * self.f(K))

    def __pow__(self, n):
        return Filter(lambda K: self.f(K)**n)

    def __truediv__(self, other):
        if isinstance(other, Number):
            return Filter(lambda K: self.f(K) / other)

    def __add__(self, other):
        return Filter(lambda K: self.f(K) + other.f(K))

    def __invert__(self):
        return Filter(lambda K: self.f(K).conj())

    def abs(self, B, P):
        return np.sqrt(self.cc(B, P, self))

    def cc(self, B, P, other):
        return (~self * other * P)(B.K).sum().real / B.size * B.res**2

    def cf(self, B, other):
        return ((~self)(B.K) * other).sum().real / B.size * B.res**2

class Identity(Filter):
    def __init__(self):
        Filter.__init__(self, lambda K: 1)

class Zero(Filter):
    def __init__(self):
        Filter.__init__(self, lambda K: 0)

def _correlation_matrix(B, P, H):
    """create a cross-correlation matrix from the set of constraint
    filters <H>"""
    m = len(H)
    M = np.zeros(shape = [m,m], dtype='float64')

    for i in range(m):
        for j in range(i + 1):
            M[i, j] = H[i].cc(B, P, H[j])
            M[j, i] = M[i, j]

    return np.matrix(M)

def _correlation_vector(B, H, f):
    """create the correlation vector for a set of constraints <H>
    and a single realisation <f>"""
    m = len(H)
    V = np.zeros(shape = [m], dtype='complex64')

    for i in range(m):
        V[i] = H[i].cf(B, f)

    return V

def dot(H, G):
    return sum((h*g for h, g in zip(H, G)), Zero())

class CGRF:
    def __init__(self, B, P, H):
        self.n = len(H)
        self.B = B
        self.H = H
        self.P = P
        self.Q = _correlation_matrix(B, P, H)

    def action(self, g):
        return np.dot(g, np.linalg.solve(self.Q, g))

    def p_value(self, g = None):
        if g == None: g = self.g
        return _chi2.cdf(self.action(g), self.n)

    def compute_field(self, g):
        zeta = np.linalg.solve(self.Q, g)
        return (dot(self.H, zeta) * self.P)(self.B.K)

    def set_coefficients(self, g):
        self.g = g

    def generate_noise(self, seed=None):
        if seed != None:
            random.seed(seed)
        self.wn = random.normal(0, 1, self.B.shape)
        self.f_r = fft.fftn(self.wn) \
            * np.sqrt(self.P(self.B.K)) \
            * Cutoff(self.B)(self.B.K)

        v = (self.f_r.conj() * self.f_r).sum() / self.B.size**2
        self.f_r /= np.sqrt(v)

        self.g_r = _correlation_vector(self.B, self.H, self.f_r).real

    def unconstrained_field(self, T = Identity()):
        return fft.ifftn(self.f_r * T(self.B.K)).real

    def mean_field(self, T = Identity()):
        f_m = self.compute_field(self.g)
        return fft.ifftn(f_m * T(self.B.K)).real

    def constrained_field(self, T = Identity()):
        f_c = self.compute_field(self.g - self.g_r)
        return fft.ifftn((self.f_r + f_c) * T(self.B.K)).real

    def triplet(self, T = Identity()):
        return self.unconstrained_field(T), self.mean_field(T), self.constrained_field(T)

def _scale_filter(B, t):
    """returns discrete scale space filter, take care with units:
        [res] = Mpc / pixel, [k] = 1 / Mpc, [t] = Mpc**2"""
    def f(K):
        return reduce(
            lambda x, y: x*y,
            (np.exp(t / B.res**2 * (np.cos(k * B.res) - 1)) for k in K))
    return f

class Scale(Filter):
    def __init__(self, B, sigma):
        Filter.__init__(self, _scale_filter(B, sigma**2))

def _K_dot(X, K):
    return sum(X[i]*K[i] for i in range(len(X)))

class Pos(Filter):
    def __init__(self, x):
        Filter.__init__(self, lambda K: np.exp(-1j * _K_dot(x, K)))

class D(Filter):
    def __init__(self, n):
        def d(i):
            return Filter(lambda K: -1j * K[i])

        A = [d(int(i)-1) for i in str(n)]
        Filter.__init__(self, reduce(operator.mul, A))

def _K_pow(k, n):
    """raise |k| to the <n>-th power safely"""
    save = np.seterr(divide = 'ignore')
    a = np.where(k == 0, 0, k**n)
    np.seterr(**save)
    return a

class Power_law(Filter):
    def __init__(self, n):
        Filter.__init__(self, lambda K: _K_pow((K**2).sum(axis=0), n/2))

class Cutoff(Filter):
    def __init__(self, B):
        Filter.__init__(self, lambda K: np.where((K**2).sum(axis=0) <= B.k_max**2, 1, 0))

class Potential(Filter):
    def __init__(self):
        Filter.__init__(self, lambda K: -_K_pow((K**2).sum(axis=0), -1.))

def make_peak(B, P, S):
    A = [Identity(), D(1), D(2), D(11), D(22), D(12)]
    return [(a * S) / (a * S).abs(B, P) for a in A]

def Multi_Gaussian(M):
    d = M.shape[0]
    A = ((2*np.pi)**d * np.linalg.det(M))**(-1/2)

    def f(x):
        S = np.dot(x, np.linalg.solve(M, x)) / 2
        return A * np.exp(-S)

    return f

def fix(f, args, free):
    """Take a function of ndarray, give default values and
    enumerate the free variables. Returns a new function that
    takes an ndarray of the same size as len(free)."""

    def g(x):
        a = args.copy() ; a[free] = x
        return f(a)

    return g

def garfield(B, P, T = Identity(), seed = None):
    if seed != None:
        random.seed(seed)
    wn = random.normal(0, 1, B.shape)
    f  = fft.ifftn(fft.fftn(wn) * np.sqrt(P(B.K))).real
    #f /= f.std()
    return fft.ifftn(fft.fftn(f) * T(B.K)).real

# vim:ts=4:sw=4:tw=80
