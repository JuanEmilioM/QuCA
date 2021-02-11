import numpy as np
from numpy import cos, sin, exp, pi
from numpy.random import uniform
from pandas import DataFrame
import matplotlib.pyplot as plt
import qutip as qip
from qutip.qip.gates import rotation, gate_expand_1toN
from qutip.qip.operations import cnot
from qutip.states import basis
from qutip import tensor
from qutip.metrics import fidelity

eps = 0.7
Re = eps    # reeward ratio
P = 1./eps  # punishment ratio

t_e = uniform(0,pi)     # random theta angle
p_e = uniform(0,2*pi)   # random phi angle
N = 200   # number of iterations

# operators
X = qip.sigmax()    # x-Pauli operator
Z = qip.sigmaz()    # z-Pauli operator
P_0 = basis(2,0)*basis(2,0).dag()   # projector onto eigenspace spanned by |0>
P_0 = gate_expand_1toN(P_0,2,0)
I = qip.qeye(2)   # identity operator
Ucnot = cnot(control=1, target=0)   # cnot gate

# initial conditions
Delta = 4*pi    # exploration range
U = I
u = I

# initial states
A = basis(2,0)  # agent in the state |0>
E = basis(2,0)*cos(t_e/2) + basis(2,1)*sin(t_e/2)*exp(1j*p_e)   # environment state

F_hist = [fidelity(E,A)]     # list to store fidelities
D_hist = [Delta/pi]     # list to store deltas

# begin iteration
for k in range(2, N+1):
    E_bar = U.dag()*E

    R = basis(2,0)  # register in the state |0>
    R_E = Ucnot*tensor(R,E_bar)  # policy
    p = P_0.matrix_element(R_E,R_E)     # p = <R_E|P_0|R_E> register measurement
    x = uniform()   # a random number to simulate the measurement result

    # this step is made for the sake of clarity
    if x < p.real:
        m=0
    else:
        m=1

    if m==0:
        Delta *= Re
    else:
        alpha = uniform(-0.5, 0.5)*Delta
        beta = uniform(-0.5, 0.5)*Delta
        u = rotation(Z, alpha)*rotation(X, beta)

        # rotation of Pauli operators
        Z = u*Z*u.dag()
        X = u*X*u.dag()

        Delta *= P
        U = u*U
        A = u*A

    if Delta > 4*pi:
        Delta = 4*pi

    f = fidelity(E, A)
    F_hist.append(f)
    D_hist.append(Delta/pi)
# end for

# save all to a csv file
x = np.arange(1,N+1)
DataFrame(data={'Iteration': x, 'Fidelity': F_hist, 'Delta': D_hist}).to_csv('data.csv', index=False)
