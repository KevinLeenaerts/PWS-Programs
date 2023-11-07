import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x1 = 20

t1 = 0
t2 = 2

def q(t):
    return -0.5 *  pow(t, 2) + x1

x2 = q(t2)

def q_prime(t):
    return -t

def eta(t):
    return -(2 * (t - t1) / (t2 - t1) - 1)**2 + 1

def eta_prime(t):
    return -8*(t-t1)/pow(t2-t1, 2)+4/(t2-t1)

def q_bar(t, omega):
    return q(t) + omega * eta(t)
    
def q_bar_prime(t, omega):
    return q_prime(t) + omega * eta_prime(t)

def L(q, q_prime):
    return 0.5 * pow(q_prime, 2) - q

t_data = np.linspace(t1, t2, 1000)

omegas = [-1, 0, 1]

fig1 = plt.figure()
plt.title("Trajectories")
plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.scatter([t1, t2], [x1, x2])

for omega in omegas:
    linestyle = "dotted"
    
    if (omega != 0.):
        linestyle = "solid"
        
    plt.plot(t_data, q_bar(t_data, omega), linestyle=linestyle)

fig2 = plt.figure()
plt.title("L field")
plt.ylabel(r"$\dot{x} (ms^{-1})$")
plt.xlabel("Height (m)")

for omega in omegas:
    linestyle = "dotted"
    
    if (omega != 0.):
        linestyle = "solid"
        
    plt.plot(q_bar(t_data, omega), q_bar_prime(t_data, omega), linestyle=linestyle)

leftX, rightX = plt.xlim()
leftY, rightY = plt.ylim()
plt.xlim(leftX, rightX)
plt.ylim(leftY, rightY)

y, x = np.meshgrid(np.linspace(leftY, rightY, 100), np.linspace(leftX, rightX, 100))
z = L(-x, 0.5 * pow(y, 2))
z = z[:-1, :-1]
plt.pcolormesh(x, y, z)
plt.colorbar()

fig3 = plt.figure()
plt.title("L in trajectories")
plt.ylabel("L (Js)")
plt.xlabel("Time (s)")

for omega in omegas:
    linestyle = "dotted"
    
    if (omega != 0.):
        linestyle = "solid"
        
    plt.plot(t_data, L(q_bar(t_data, omega), q_bar_prime(t_data, omega)), linestyle=linestyle)

fig4 = plt.figure()
plt.title("Action of trajectories")
plt.ylabel("S (J)")
plt.xlabel("Omega ")

heights = []
for omega in omegas:
    heights.append(np.sum(L(q_bar(t_data, omega), q_bar_prime(t_data, omega))))
  
plt.bar(omegas, heights)

fig1.savefig('output/trajectories.pdf', format='pdf', bbox_inches='tight')
fig2.savefig('output/lagrange_field.pdf', format='pdf', bbox_inches='tight')
fig3.savefig('output/lagrange_trajectories.pdf', format='pdf', bbox_inches='tight')
fig4.savefig('output/trajectories_actions.pdf', format='pdf', bbox_inches='tight')

plt.show()