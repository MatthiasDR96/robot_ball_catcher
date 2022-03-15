'''
Dit script zet de opgenomen informatie om in X,Y,Z ruimtecoördinaten.
Vervolgens wordt de opgenomen informatie gevisualiseerd in een 2d plot en een 3d scatterplot.
'''

# Imports
import os
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import numpy
from robot_ball_catcher.Camera import Camera
from Viewer import Viewer
from image_processing import *

# Get HSV calibration params
datapath = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '\data'

# Get camera calibration params
mtx = numpy.load(datapath + '\intrinsics.npy')
dist = numpy.load(datapath + '\distortion.npy')
ext = numpy.load(datapath + '\extrinsics.npy')
input = numpy.load(datapath + '\prerecording.npy')

# print de gegevens van de opname
print('input:\n', input)
length = round(len(input[:, :] / 4))
print('frames:\n', length)

# Make world coordinate matrix
worldmatrix = numpy.zeros((length, 4))

# Loop
for i in range(length):
    time = input[i, 0]
    u_ball = input[i, 1]
    v_ball = input[i, 2]
    z_ball = input[i, 3]
    if 1:
        xcam, ycam, zcam = func.intrinsictrans((u_ball, v_ball), z_ball, mtx)
        xworld, yworld, zworld = func.extrinsictrans(z_ball, xcam, ycam, zcam, ext)
        worldmatrix[i, 0] = time
        worldmatrix[i, 1] = xworld
        worldmatrix[i, 2] = yworld
        worldmatrix[i, 3] = zworld

# Save world coordinate matrix
numpy.save(datapath + '\worldmatrix.npy', worldmatrix)

# Make canvas
fig1 = plt.figure()
fig1.tight_layout()
ax1 = fig1.add_subplot(311)
ax2 = fig1.add_subplot(312)
ax3 = fig1.add_subplot(313)

# Plot data
ax1.plot(worldmatrix[:, 0], worldmatrix[:, 1])
ax2.plot(worldmatrix[:, 0], worldmatrix[:, 2])
ax3.plot(worldmatrix[:, 0], worldmatrix[:, 3])

# Set canvas data
ax1.set_title('X')
ax1.set_xlabel('time (ms)')
ax1.set_ylabel('coordinate (mm)')
ax1.set_ylim(-1000, 1000)
ax2.set_title('Y')
ax2.set_xlabel('time (ms)')
ax2.set_ylabel('coordinate (mm)')
ax2.set_ylim(-1000, 1000)
ax3.set_title('Z')
ax3.set_xlabel('time (ms)')
ax3.set_ylabel('coordinate (mm)')
ax3.set_ylim(0, 600)

# Show
plt.show()