from DoublePendulum import DoublePendulum
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import hsv_to_rgb
import numpy as np
import datetime
import math

time = 50
traceTime = 1
FPS = 24
frames = time * FPS
traceFrames = traceTime * FPS
t = np.linspace(0, time, frames)

theta = 1
thetaVel = 10**-10

x1_data = []
y1_data = []
x2_data = []
y2_data = []

fig, (ax1, ax2, ax3) = plt.subplots(3)
lines = []
traceLines = []
regressionLines = []

x1, y1, x2, y2 = DoublePendulum.generateData(1, 1, 2, 1, theta, -thetaVel, 0, 0, t)
x1_data.append(x1)
y1_data.append(y1)
x2_data.append(x2)
y2_data.append(y2)
line, = ax1.plot([], [], lw=3, markersize=8)
traceLine, = ax1.plot([], [], lw=3, markersize=8, alpha=0.2)
lines.append(line)
traceLines.append(traceLine)

x1, y1, x2, y2 = DoublePendulum.generateData(1, 1, 2, 1, theta, thetaVel, 0, 0, t)
x1_data.append(x1)
y1_data.append(y1)
x2_data.append(x2)
y2_data.append(y2)
line, = ax1.plot([], [], lw=3, markersize=8)
traceLine, = ax1.plot([], [], lw=3, markersize=8, alpha=0.2)
lines.append(line)
traceLines.append(traceLine)

regressionData = np.hypot(np.subtract(x2_data[1], x2_data[0]), np.subtract(y2_data[1], y2_data[0]))
totalRegressionData = np.cumsum(regressionData)
regressionLine, = ax2.plot(t, regressionData, lw=3, markersize=8)
regressionCurrentTime = ax2.axvline(0)

totalRegressionLine, = ax3.plot(t, totalRegressionData, lw=3, markersize=8)
totalRegressionCurrentTime = ax3.axvline(0)

def animate(frame):
    for i in range(len(lines)):
        lines[i].set_data([0, x1_data[i][frame], x2_data[i][frame]], [0, y1_data[i][frame], y2_data[i][frame]])
        
        trace_x_data = x2_data[i][np.clip(frame-traceFrames, 0, frames):frame]
        trace_y_data = y2_data[i][np.clip(frame-traceFrames, 0, frames):frame]
        traceLines[i].set_data(trace_x_data, trace_y_data)
        
    regressionCurrentTime.set_xdata(frame/frames * time)
    totalRegressionCurrentTime.set_xdata(frame/frames * time)
    if (frame % 10 == 0):
        print("Generating frames: " + str(100 * (frame / frames)) + "%", end='\r')
       
    return lines[0], lines[1]

ymax = max(np.amax(y2_data), 0) * 1.1
ymin = np.amin(y2_data) * 1.1
xmax = np.amax(np.abs(x2_data)) * 1.1
ax1.set_ylim(ymin, ymax)
ax1.set_xlim(-xmax, xmax)
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)

ax2.set_ylim(np.amin(regressionData), np.amax(regressionData))
ax2.set_xlim(0, np.amax(t))
ax2.grid()

ax3.set_ylim(np.amin(totalRegressionData), np.amax(totalRegressionData))
ax3.set_xlim(0, np.amax(t))
ax3.grid()


ani = animation.FuncAnimation(fig, animate, frames=frames, interval=1/FPS * 1000)

plt.show()
ani.save("output.mp4")