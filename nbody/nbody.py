#!/usr/bin/env python3

import numpy as np
from numpy import fft, random
import gnuplot as gp
from scipy.integrate import quad

class Cosmology:
    """This object stores all relevant information wrt the background
    cosmology, parametrized by OmegaM, OmegaL and H0."""
    def __init__(self, H0, OmegaM, OmegaL):
        self.H0 = H0
        self.OmegaM = OmegaM
        self.OmegaL = OmegaL
        self.OmegaK = 1 - OmegaM - OmegaL
        self.grav_cst = 3./2 * OmegaM * H0**2
        self.factor = self._growing_mode_norm()

    def adot(self, a):
        return self.H0 * a * np.sqrt(self.OmegaL \
                + self.OmegaM * a**-3 \
                + self.OmegaK * a**-2)
    
    def _growing_mode_norm(self):
        """result D(1) = 1. d/d0 + 0.001 = 1"""
        d = self.adot(1) * quad(lambda b: self.adot(b)**(-3), 0.00001, 1)[0]
        return 0.99999/d

    def growing_mode(self, a):
        if isinstance(a, np.ndarray):
            return np.array([self.growing_mode(b) for b in a])
        elif a <= 0.001:
            return a
        else:
            return self.factor * self.adot(a)/a * quad(lambda b: self.adot(b)**(-3), 0.00001, a)[0] + 0.00001
    
    def __call__(self, a):
        return self.adot(a)

LCDM = Cosmology(68.0, 0.31, 0.69)
EdS = Cosmology(70.0, 1.0, 0.0)

def md_cic(B, X):
    f  = X - np.floor(X)
    
    rho = np.zeros(B.shape, dtype='float64')
    rho_, x_, y_ = np.histogram2d(X[:,0]%B.N, X[:,1]%B.N, bins=B.shape, 
                        range=[[0, B.N], [0, B.N]], 
                        weights=(1 - f[:,0])*(1 - f[:,1]))
    rho += rho_
    rho_, x_, y_ = np.histogram2d((X[:,0]+1)%B.N, X[:,1]%B.N, bins=B.shape, 
                        range=[[0, B.N], [0, B.N]], 
                        weights=(f[:,0])*(1 - f[:,1]))
    rho += rho_
    rho_, x_, y_ = np.histogram2d(X[:,0]%B.N, (X[:,1]+1)%B.N, bins=B.shape, 
                        range=[[0, B.N], [0, B.N]], 
                        weights=(1 - f[:,0])*(f[:,1]))
    rho += rho_
    rho_, x_, y_ = np.histogram2d((X[:,0]+1)%B.N, (X[:,1]+1)%B.N, bins=B.shape, 
                        range=[[0, B.N], [0, B.N]], 
                        weights=(f[:,0])*(f[:,1]))
    rho += rho_
    
    return rho

class Integrator:
    def run(self, a0, da, N):
        raise NotImplented

class LeapFrog(Integrator):
    def __init__(self, solver):
        self.solver = solver
        
    def run(self, a0, da, N):
        a = a0
        
        # the kick is half a step ahead
        S = self.solver(a, a + da/2)
        for i in range(N):
            S.drift(a, da)
            S.kick(a+da/2, da)
            a += da
            
        return S

class Interp2D:
    "Reasonably fast bilinear interpolation routine"
    def __init__(self, data):
        self.data = data
        self.shape = data.shape

    def __call__(self, x):
        X1 = np.floor(x).astype(int) % self.shape
        X2 = np.ceil(x).astype(int) % self.shape
        xm = x % 1.0
        xn = 1.0 - xm

        f1 = self.data[X1[:,0], X1[:,1]]
        f2 = self.data[X2[:,0], X1[:,1]]
        f3 = self.data[X1[:,0], X2[:,1]]
        f4 = self.data[X2[:,0], X2[:,1]]

        return  f1 * xn[:,0] * xn[:,1] + \
                f2 * xm[:,0] * xn[:,1] + \
                f3 * xn[:,0] * xm[:,1] + \
                f4 * xm[:,0] * xm[:,1]

def gradient_2nd_order(F, i):
	return   1./12 * np.roll(F,  2, axis=i) - 2./3  * np.roll(F,  1, axis=i) \
			+ 2./3  * np.roll(F, -1, axis=i) - 1./12 * np.roll(F, -2, axis=i)

def a2r(B, X):
    return X.transpose([1,2,0]).reshape([B.N**2, 2])

def r2a(B, x):
    return x.reshape([B.N, B.N, 2]).transpose([2,0,1])

class Solver:
    def __init__(self, B, m, C, X, P):
        self.B = B
        self.C = C
        self.X = X
        self.P = P
        self.m = m
        
        self.g = gp.Gnuplot(persist=True)
        self.g("set cbrange [0.2:50]", "set log cb", "set size square",
               "set xrange [0:{0}] ; set yrange [0:{0}]".format(B.N))
        
    def drift(self, a, da):
        adot = self.C(a)
        self.X += da * self.P / (a**2 * adot)
        
    def kick(self, a, da):
        adot  = self.C(a)
        x = self.X/self.B.res
        delta = md_cic(self.B, x) * self.m - 1.0
        self.g(gp.plot_data(gp.array(delta.T+1, "t'' w image")))
        fn = 'data/x.{0:05d}.npy'.format(int(round(a*1000)))
        np.save(fn, self.X)
        
        phi   = fft.ifftn(fft.fftn(delta) * cft.Potential()(self.B.K)).real * self.C.grav_cst / a
        
        acc_x = Interp2D(gradient_2nd_order(phi, 0))
        acc_y = Interp2D(gradient_2nd_order(phi, 1))
        acc = np.c_[acc_x(x), acc_y(x)] / self.B.res
        
        self.P -= da * acc / adot

class Zeldovich:
    def __init__(self, B_mass, B_force, C, phi):
        self.bm = B_mass
        self.bf = B_force
        self.C  = C
        
        self.u = np.array([-gradient_2nd_order(phi, 0), 
                           -gradient_2nd_order(phi, 1)]) / self.bm.res
        
    def __call__(self, a_pos, a_vel = None):
        if a_vel == None: 
            a_vel = a_pos
        
        X = a2r(self.bm, np.indices(self.bm.shape) * self.bm.res + a_pos * self.u)
        P = a2r(self.bm, a_vel * self.u)
        
        # particle mass
        m = (self.bf.N / self.bm.N)**self.bm.dim
        
        return Solver(self.bf, m, self.C, X, P)

import cft

if __name__ == "__main__":
	N = 256
	B_m = cft.Box(2, N, 50.0)

	A = 10
	seed = 4
	Power_spectrum = cft.Power_law(-0.5) * cft.Scale(B_m, 0.2) * cft.Cutoff(B_m)

	phi = cft.garfield(B_m, Power_spectrum, cft.Potential(), seed) * A
	rho = cft.garfield(B_m, Power_spectrum, cft.Scale(B_m, 0.5), seed) * A

	za = Zeldovich(B_m, cft.Box(2, N*2, B_m.L), EdS, phi)
	S = LeapFrog(za).run(0.02, 0.02, 200)

