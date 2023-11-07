import numpy as np
import sympy as smp
from scipy.integrate import odeint

t, g, m1,  m2, l1, l2 = smp.symbols('t, g, m1, m2, l1, l2')

theta1, theta2 = smp.symbols(r'\theta_1, \theta_2', cls=smp.Function)
theta1 = theta1(t)
theta2 = theta2(t)

theta1_dot = smp.diff(theta1, t)
theta2_dot = smp.diff(theta2, t)
theta1_ddot = smp.diff(theta1_dot, t)
theta2_ddot = smp.diff(theta2_dot, t)
        
L1 = smp.Eq(l1 * theta1_ddot * (m1 + m2) + m2 * l2 * theta2_ddot * smp.cos(theta1 - theta2), -g * smp.sin(theta1) * (m1 + m2) - m2 * l2 * smp.Pow(theta2_dot, 2) * smp.sin(theta1 - theta2))
L2 = smp.Eq(m2 * l1 * theta1_ddot * smp.cos(theta1 - theta2) + m2 * l2 * theta2_ddot, -g * m2 * smp.sin(theta2) + m2 * l2 * smp.Pow(theta1_dot, 2) * smp.sin(theta1 - theta2))
        
solutions = smp.solve([L1, L2], (theta1_ddot, theta2_ddot), simplify=True)

class DoublePendulum:
    
    g = 9.81
    
    def generateData(m1, m2, l1, l2, theta1, theta1_dot, theta2, theta2_dot, t):
        dz1dt_f, dz2dt_f, dthe1dt_f, dthe2dt_f = DoublePendulum.getDiffEquations()
        
        ans = odeint(lambda S, t, g, m1, m2, l1, l2, dz1dt_f, dz2dt_f, dthe1dt_f, dthe2dt_f : DoublePendulum.dSdt(S, t, g, m1, m2, l1, l2, dz1dt_f, dz2dt_f, dthe1dt_f, dthe2dt_f), y0=[theta1, theta1_dot, theta2, theta2_dot], t=t, args=(DoublePendulum.g,m1,m2,l1,l2, dz1dt_f, dz2dt_f, dthe1dt_f, dthe2dt_f))

        theta1 = ans.T[0]
        theta2 = ans.T[2]
        
        return (
            l1*np.sin(theta1),
            -l1*np.cos(theta1),
            l1*np.sin(theta1) + l2*np.sin(theta2),
            -l1*np.cos(theta1) - l2*np.cos(theta2)
        )
        
    def getDiffEquations():
        dz1dt_f = smp.lambdify((t, g, m1, m2, l1, l2, theta1, theta2, theta1_dot, theta2_dot), solutions[theta1_ddot])
        dz2dt_f = smp.lambdify((t, g, m1, m2, l1, l2, theta1, theta2, theta1_dot, theta2_dot), solutions[theta2_ddot])
        dthe1dt_f = smp.lambdify(theta1_dot, theta1_dot)
        dthe2dt_f = smp.lambdify(theta2_dot, theta2_dot)
        
        return dz1dt_f, dz2dt_f, dthe1dt_f, dthe2dt_f
        
    def dSdt(S, t, g, m1, m2, l1, l2, dz1dt_f, dz2dt_f, dthe1dt_f, dthe2dt_f):
        the1, z1, the2, z2 = S
        
        return [
            dthe1dt_f(z1),
            dz1dt_f(t, g, m1, m2, l1, l2, the1, the2, z1, z2),
            dthe2dt_f(z2),
            dz2dt_f(t, g, m1, m2, l1, l2, the1, the2, z1, z2),
        ]   