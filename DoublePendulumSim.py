from DoublePendulum import DoublePendulum
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import hsv_to_rgb
import numpy as np
import datetime

time = 30
traceTime = 1
FPS = 24
frames = time * FPS
traceFrames = traceTime * FPS;

t = np.linspace(0, time, frames)

x1_data = []
y1_data = []
x2_data = []
y2_data = []

lines = []
outputTraceLines = []

fig, ax = plt.subplots(1,1)
# ax.set_facecolor('k')

startValue = 2
endValue = 2.0001
pendula = 300
colors = plt.cm.jet(np.linspace(0, 1, pendula))

for index, theta_1 in enumerate(np.linspace(startValue, endValue, pendula)):
    index_percentage = (theta_1 - startValue) / (endValue - startValue)
    
    x1, y1, x2, y2 = DoublePendulum.generateData(1, 1, 2, 1, theta_1, 0, 0, 0, t)
    x1_data.append(x1)
    y1_data.append(y1)
    x2_data.append(x2)
    y2_data.append(y2)

    line, = plt.plot([], [], color=colors[index], lw=3, markersize=8)
    lines.append(line)
    
    line, = plt.plot([], [], color=colors[index], lw=3, markersize=8, alpha=0.1)
    outputTraceLines.append(line)
    
    print("Generating data: " + str(100 * index_percentage) + "%", end='\r')

def animate(frame):
    for i in range(len(lines)):    
        lines[i].set_data([0, x1_data[i][frame], x2_data[i][frame]], [0, y1_data[i][frame], y2_data[i][frame]])
        
        trace_x_data = x2_data[i][np.clip(frame-traceFrames, 0, frames):frame]
        trace_y_data = y2_data[i][np.clip(frame-traceFrames, 0, frames):frame]
        outputTraceLines[i].set_data(trace_x_data, trace_y_data)
        
    if (frame % 10 == 0):
        print("Generating frames: " + str(100 * (frame / frames)) + "%", end='\r')
       
ymax = np.amax(y2_data) * 1.1
ymin = np.amin(y2_data) * 1.1
xmax = np.amax(np.abs(x2_data)) * 1.1
ax.set_ylim(ymin, ymax)
ax.set_xlim(-xmax, xmax)

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

ani = animation.FuncAnimation(fig, animate, frames=frames, interval=1/FPS * 1000)
# ani.save("output.mp4")
plt.show()