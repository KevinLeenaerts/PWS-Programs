from DoublePendulum import DoublePendulum
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import math
from matplotlib import cm
import time as clock

resolution = 20
thetaRange = math.pi - 1

time = 120
FPS = 25
frames = time * FPS
timeResolution = time * FPS

dtheta = 10**-10

t = np.linspace(0, time, timeResolution)

theta1_data = np.linspace(-thetaRange, thetaRange, resolution)
theta2_data = np.linspace(-thetaRange, thetaRange, resolution)

regressions = np.zeros((len(theta1_data), len(theta2_data)), dtype=object)

lastResult = 0
gain = 0.3

for index1, theta1 in enumerate(theta1_data):
    startTime = clock.time()
    for index2, theta2 in enumerate(theta2_data):
        x1_1, y1_1, x2_1, y2_1 = DoublePendulum.generateData(2, 1, 2, 1, theta1 - dtheta/2, 0, theta2 - dtheta / 2, 0, t)
        x1_2, y1_2, x2_2, y2_2 = DoublePendulum.generateData(2, 1, 2, 1, theta1 + dtheta/2, 0, theta2 + dtheta / 2, 0, t)

        distances = np.hypot(np.subtract(x2_2, x2_1), np.subtract(y2_2, y2_1))
        regression_data = np.cumsum(distances)
        
        regressions[index2][index1] = regression_data
        
    dt = clock.time() - startTime
    
    if (lastResult == 0):
        lastResult = dt
        
    filteredDt = gain * dt + (1 - gain) * lastResult
    lastResult = filteredDt
    
    framesRemaining = len(theta1_data) - index1
    timeRemaining = framesRemaining * filteredDt
    timeDelta = clock.strftime("%H:%M:%S", clock.gmtime(timeRemaining))
    
    print("Generating data: " + str(index1 / len(theta1_data) * 100) + "%, Remaining: " + timeDelta, end='\r')


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X, Y = np.meshgrid(theta1_data, theta2_data)
Z = np.array([[sub_array[0] for sub_array in row] for row in regressions])
mesh = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

lastResult = 0
def animate(frame):
    dt = clock.time() - startTime
    sliced = np.array([[sub_array[frame] for sub_array in row] for row in regressions])
    
    ax.clear()
    ax.plot_surface(X, Y, sliced, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    
    framesRemaining = len(theta1_data) - index1
    timeRemaining = framesRemaining * filteredDt
    timeDelta = clock.strftime("%H:%M:%S", clock.gmtime(timeRemaining))
    
    print("Generating frames: " + str(frame / frames * 100) + "%, Remaining: " + timeDelta, end='\r')
    
ax.set_zlim(0, np.amax([[sub_array[resolution - 1] for sub_array in row] for row in regressions]))
ax.set_xlabel(r"$\theta_1$")
ax.set_ylabel(r"$\theta_2$")
cbar = fig.colorbar(mesh)
cbar.set_label("Regressie (rad*s)", rotation=270)

ani = animation.FuncAnimation(fig, animate, frames=frames, interval=1/FPS * 1000)
ani.save("regression.mp4")
# plt.show()