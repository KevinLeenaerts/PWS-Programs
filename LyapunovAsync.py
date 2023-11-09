import numpy as np
from numpy import sin, cos
import scipy
from math import log
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time as clock
import asyncio

G = 9.81
L1 = 1.0
L2 = 1.0
t_stop = 30
M_start = 0.5
M_stop = 15
M_step = 0.5
angles_res = 10
dt = 0.02

gain = 0.3
lastResult = 0

def derivs(state, t, M1, M2):
    dydx = np.zeros_like(state)
    
    dydx[0] = state[1]

    delta = state[2] - state[0]
    den1 = (M1 + M2) * L1 - M2 * L1 * cos(delta) * cos(delta)
    dydx[1] = ((M2 * L1 * state[1] * state[1] * sin(delta) * cos(delta)
                + M2 * G * sin(state[2]) * cos(delta)
                + M2 * L2 * state[3] * state[3] * sin(delta)
                - (M1 + M2) * G * sin(state[0]))
               / den1)

    dydx[2] = state[3]

    den2 = (L2 / L1) * den1
    dydx[3] = ((- M2 * L2 * state[3] * state[3] * sin(delta) * cos(delta)
                + (M1 + M2) * G * sin(state[0]) * cos(delta)
                - (M1 + M2) * L1 * state[1] * state[1] * sin(delta)
                - (M1 + M2) * G * sin(state[2]))
               / den2)
    
    return dydx

t = np.arange(0, t_stop, dt)

def get_theta1_array(theta1, theta1_dot, theta2, theta2_dot, M1, M2):
    state = [theta1, theta1_dot, theta2, theta2_dot]
    results = scipy.integrate.odeint(lambda S, t : derivs(S, t, M1, M2), state, t)
    return results[:, 0]

async def trajectory_difference(theta1, theta2, m1, m2, dtheta):
    trajectory1 = get_theta1_array(theta1, 0, theta2, 0, m1, m2)
    trajectroy2 =  get_theta1_array(theta1 + dtheta, 0, theta2 + dtheta, 0, m1, m2)
    return np.abs(np.subtract(trajectory1, trajectroy2))

async def get_lyapunov_values(differences):
    lyapunov = [0]
    
    for N in range(1, len(differences)):
        exponent = (1 / N) * log(differences[N] / differences[0])
        lyapunov.append(exponent)
        
    return lyapunov

async def set_result_row(result_row, m1, m2, index2):
    lyapunov_sum = 0
        
    for angle in angles_array:
        differences = await asyncio.gather(trajectory_difference(angle, angle, m1, m2, 0.01))
        maximum_lyapunov = max(await get_lyapunov_values(differences))
        lyapunov_sum += maximum_lyapunov
            
    result_row[index2] = lyapunov_sum / len(angles_array)

M_array = np.arange(M_start, M_stop, M_step)
angles_array = np.linspace(0, 0.5 * math.pi, angles_res)
result_matrix = np.zeros((len(M_array), len(M_array)))

event_loop = asyncio.new_event_loop()
asyncio.set_event_loop(event_loop)

for index1, m1 in enumerate(M_array):
    startTime = clock.time()
    
    result_row = np.zeros(len(M_array))
    coroutines = [set_result_row(result_row, m1, m2, index2) for index2, m2 in enumerate(M_array)]
    event_loop.run_until_complete(asyncio.gather(*coroutines))
    
    result_matrix[index1] = result_row
    
    # Print estimated time
    dt = clock.time() - startTime
    
    if (lastResult == 0):
        lastResult = dt
        
    filteredDt = gain * dt + (1 - gain) * lastResult
    lastResult = filteredDt
    
    framesRemaining = len(M_array) - index1
    timeRemaining = framesRemaining * filteredDt
    timeDelta = clock.strftime("%H:%M:%S", clock.gmtime(timeRemaining))
    
    print("Generating data: " + str(100 - (framesRemaining / len(M_array)) * 100) + "%, Remaining: " + timeDelta, end='\r')
    
final_result = pd.DataFrame(result_matrix, M_array, M_array)

plt.subplots(1)
heat_map = sns.heatmap(final_result, annot=False, square=True, cbar_kws={'label': 'Lyapunov exponent'})
heat_map.invert_yaxis()
heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=90, horizontalalignment='center')
plt.title("Average maximum Lyapunov exponent for different mass combinations")
plt.xlabel(r"$l_2$(kg)")
plt.ylabel(r"$l_1$ (kg)")
plt.show()