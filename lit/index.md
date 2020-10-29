---
title: 2D particle-mesh n-body code
author: Johan Hidding
---

This is a particle-mesh n-body code for cosmological n-body simulations. This code has several uses. For many methods of analysis in cosmology it can be very helpful to have a 2D sample available to test them with. This code is very nice to play around with for students, since it is written in 100% Python. Lastly, having 2D simulations can give a great deal of insight.

``` {.python file=nbody/nbody.py}
<<imports>>

<<cosmology>>
<<mass-deposition>>
<<interpolation>>
<<integrator>>
<<solver>>
<<initialization>>
<<main>>
```

Some things we will definitely need:

``` {.python #imports}
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad
from .cft import Box
from . import gnuplot as gp

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Callable
from functools import partial
```

# The Box
The `Box` class contains all information about the simulation box: mainly the size in pixels and the physical size it represents. All operations will assume periodic boundary conditions. In `numpy` this is achieved by using `np.roll` to shift a grid along a given axis.

# Cosmology
The background cosmology is described by a function giving the scale function $a(t)$ as a function of time. In standard Big Bang cosmology this scale function is computed from three parameters (ignoring baryons): $H_0$ the Hubble expansion rate at $t_0$ ($t_0$ being now), $\Omega_{m}$ the matter density expressed as a fraction of the critical density, and $\Omega_{\Lambda}$ the dark energy (cosmological constant) component, again expressed as a fraction of the critical density.

``` {.python #cosmology}
@dataclass
class Cosmology:
    H0 : float
    OmegaM : float
    OmegaL : float

    <<cosmology-methods>>
```

From these parameters, we can compute the curvature, 

$$\Omega_k = 1 - \Omega_{m} - \Omega_{\Lambda},$${#eq:curvature}

``` {.python #cosmology-methods}
@property
def OmegaK(self):
    return 1 - self.OmegaM - self.OmegaL
```

and the gravitational constant,

$$G = \frac{3}{2} \Omega_{m} H_0^2.$${#eq:grav-const}

``` {.python #cosmology-methods}
@property
def G(self):
    return 3./2 * self.OmegaM * self.H0**2
```

We can now express the expansion factor as an initial value problem,

$$\partial_t a(a) = H_0 a \sqrt{\Omega_{\Lambda} + \Omega_{m} a^{-3} + \Omega_{k} a^{-2}}.$$

``` {.python #cosmology-methods}
def da(self, a):
    return self.H0 * a * np.sqrt(
              self.OmegaL \
            + self.OmegaM * a**-3 \
            + self.OmegaK * a**-2)
```

For all cases that we're interested in, we can integrate this equation directly to obtain the growing mode solution. We cannot start the integration from $a=0$, but in the limit of $a \to 0$, we have that $\D_{+} \approx a$.

``` {.python #cosmology-methods}
def growing_mode(self, a):
    if isinstance(a, np.ndarray):
        return np.array([self.growing_mode(b) for b in a])
    elif a <= 0.001:
        return a
    else:
        return self.factor * self.adot(a)/a \
            * quad(lambda b: self.adot(b)**(-3), 0.00001, a)[0] + 0.00001
```

Using this, we can define two standard cosmologies, $\Lambda$CDM and Einstein-de Sitter.

``` {.python #cosmology}
LCDM = Cosmology(68.0, 0.31, 0.69)
EdS = Cosmology(70.0, 1.0, 0.0)
```

# Mass deposition
To do the mass deposition, that is, convert the position of particles into a 2D mesh of densities, we use the cloud-in-cell method. Every particle is smeared out over its four nearest neighbours, weighted by the distance to each neighbour. This principle is similar (but inverse) to a linear interpolation scheme: we compute the integer index of the grid-cell the particle belongs to, and use the floating-point remainder to compute the fractions in all the four neighbours. In this case however, we abuse the `histogram2d` function in `numpy` to do the interpolation for us. 

``` {.python #mass-deposition}
def md_cic(B: Box, X: np.ndarray) -> np.ndarray:
    """Takes a 2*M array of particle positions and returns an array of shape
    `B.shape`. The result is a density field computed by cloud-in-cell method."""
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
```

# Interpolation
To read a value from a grid, given a particle position, we need to interpolate. This routine performs linear interpolation on the grid.

``` {.python #interpolation}
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
```

# Leap-frog integrator
The Leap-frog method is a generic method for solving Hamiltonian systems. We divide the integration into a *kick* and *drift* stage. In the Leap-frog method, the kicks happen in between the drifts.

It is nice to write this part of the program in a formal way. We define an abstract `Vector` type, which will store the position and momentum variables.

``` {.python #integrator}
class VectorABC(ABC):
    @abstractmethod
    def __add__(self, other: Vector) -> Vector:
        raise NotImplementedError

    @abstractmethod
    def __rmul__(self, other: float) -> Vector:
        raise NotImplementedError

VectorABC.register(np.ndarray)

Vector = TypeVar("Vector", bound=VectorABC)
```

Given a `Vector` type, we define the `State` to be the combination of `position`, `momentum` and `time` (due to FRW dynamics on the background, the system is time dependent).

``` {.python #integrator}
@dataclass
class State(Generic[Vector]):
    time : float
    position : Vector
    momentum : Vector

    <<state-methods>>
```

We may manipulate the `State` in three ways: `kick`, `drift` or `wait`. *Kicking* the state means changing the momentum by some amound, given by the momentum equation. *Drifting* means changing the position following the position equation. *Waiting* simply sets the clock forward.

``` {.python #state-methods}
def kick(self, dt: float, h: HamiltonianSystem[Vector]) -> State[Vector]:
    self.momentum += dt * h.momentumEquation(self)
    return self

def drift(self, dt: float, h: HamiltonianSystem[Vector]) -> State[Vector]:
    self.position += dt * h.positionEquation(self)
    return self

def wait(self, dt: float) -> State[Vector]:
    self.time += dt
    return self
```

The combination of a position and momentum equation is known as a Hamiltonian system:

``` {.python #integrator}
class HamiltonianSystem(ABC, Generic[Vector]):
    @abstractmethod
    def positionEquation(self, s: State[Vector]) -> Vector:
        raise NotImplementedError

    @abstractmethod
    def momentumEquation(self, s: State[Vector]) -> Vector:
        raise NotImplementedError
```

A `Solver` is a function that takes a Hamiltonian system, an initial state and returns a final state. A `Stepper` translates one state to the next. The `HaltingCondition` is function of the state that determines when to stop integrating.

``` {.python #integrator}
Solver = Callable[[HamiltonianSystem[Vector], State[Vector]], State[Vector]]
Stepper = Callable[[State[Vector]], State[Vector]]
HaltingCondition = Callable[[State[Vector]], bool]
```

Now we have the tools in hand to give a very consise definition of the Leap-frog integrator, namely: `kick dt` -- `wait dt/2` -- `drift dt` -- `wait dt/2`.

``` {.python #integrator}
def leap_frog(dt: float, h: HamiltonianSystem[Vector], s: State[Vector]) -> State[Vector]:
    return s.kick(dt, h).wait(dt/2).drift(dt, h).wait(dt/2)
```

From the integrator we can construct a `Stepper` function (`step = partial(leap_frog, dt, system)`), that we can iterate until completion.

``` {.python #integrator}
def iterate_step(step: Stepper, halt: HaltingCondition, init: State[Vector]) -> State[Vector]:
    while not halt(init):
        init = step(init)
    return init
```

# Poisson solver
Now for the hardest bit. We need to solve the Poisson equation.

``` {.python #solver}
def a2r(B, X):
    return X.transpose([1,2,0]).reshape([B.N**2, 2])

def r2a(B, x):
    return x.reshape([B.N, B.N, 2]).transpose([2,0,1])

class PoissonVlasov(HamiltonianSystem[np.ndarray]):
    def __init__(self, box, cosmology, particle_mass):
        self.box = box
        self.cosmology = cosmology
        self.particle_mass = particle_mass
        self._g = gp.Gnuplot(persist=True)
        self._g("set cbrange [0.2:50]", "set log cb", "set size square",
                "set xrange [0:{0}] ; set yrange [0:{0}]".format(box.N))
        self._g("set term x11")
        # self._g(gp.default_palette)

    def positionEquation(self, s: State[np.ndarray]) -> np.ndarray:
        a = s.time
        da = self.cosmology.da(a)
        return s.momentum / (s.time**2 * da)

    def momentumEquation(self, s: State[np.ndarray]) -> np.ndarray:
        a = s.time
        da = self.cosmology.da(a)
        x_grid = s.position / self.box.res
        delta = md_cic(self.box, x_grid) * self.particle_mass - 1.0
        self._g(gp.plot_data(gp.array(delta.T+1, "t'' w image")))
        delta_f = np.fft.fftn(delta)
        kernel = cft.Potential()(self.box.K)
        phi = np.fft.ifftn(delta_f * kernel).real * self.cosmology.G / a
        acc_x = Interp2D(gradient_2nd_order(phi, 0))
        acc_y = Interp2D(gradient_2nd_order(phi, 1))
        acc = np.c_[acc_x(x_grid), acc_y(x_grid)] / self.box.res
        return -acc / da
```

# The Zeldovich Approximation
To bootstrap the simulation, we need to create a set of particles and assign velocities. This is done using the Zeldovich Approximation.

``` {.python #initialization}
class Zeldovich:
    def __init__(self, B_mass: Box, B_force: Box, cosmology: Cosmology, phi: np.ndarray):
        self.bm = B_mass
        self.bf = B_force
        self.cosmology  = cosmology
        self.u = np.array([-gradient_2nd_order(phi, 0),
                           -gradient_2nd_order(phi, 1)]) / self.bm.res

    def state(self, a_init: float) -> State[np.ndarray]:
        X = a2r(self.bm, np.indices(self.bm.shape) * self.bm.res + a_init * self.u)
        P = a2r(self.bm, a_init * self.u)
        return State(time=a_init, position=X, momentum=P)

    @property
    def particle_mass(self):
        return (self.bf.N / self.bm.N)**self.bm.dim
```

``` {.python #main}
if __name__ == "__main__":
    from . import cft

    N = 256
    B_m = Box(2, N, 50.0)

    A = 10
    seed = 4
    Power_spectrum = cft.Power_law(-0.5) * cft.Scale(B_m, 0.2) * cft.Cutoff(B_m)

    phi = cft.garfield(B_m, Power_spectrum, cft.Potential(), seed) * A
    rho = cft.garfield(B_m, Power_spectrum, cft.Scale(B_m, 0.5), seed) * A

    force_box = cft.Box(2, N*2, B_m.L)
    za = Zeldovich(B_m, force_box, EdS, phi)
    state = za.state(0.02)
    system = PoissonVlasov(force_box, EdS, za.particle_mass)
    stepper = partial(leap_frog, 0.02, system)
    iterate_step(stepper, lambda s: s.time > 4.0, state)
```
