# Imports
import os
import numpy
from robot_ball_catcher.Camera import Camera
from robot_ball_catcher.Viewer import Viewer
from robot_ball_catcher.image_processing import *

# Get HSV calibration params path
hsvfile = numpy.load('./data/hsv.npy')

# Get HSV calibration params
lower_color = numpy.array([hsvfile[0], hsvfile[2], hsvfile[4]])
upper_color = numpy.array([hsvfile[1], hsvfile[3], hsvfile[5]])

# Get camera calibration params
mtx = numpy.load('./data/intrinsics.npy')
dist = numpy.load('./data/distortion.npy')
ext = numpy.load('./data/extrinsics.npy')
rvecs = numpy.load('./data/extrinsic_rvecs.npy')
tvecs = numpy.load('./data/extrinsic_tvecs.npy')

# Window settings
window_name = 'XYZ ball detection'
window_b, window_h = 1920, 1080

# minimum straal van de bal contour
minradius = 30

# Get camera object
cam = Camera()

# Get Viewer object
viewer = Viewer()

# Set viewer
viewer.set_window(window_name, window_b, window_h)

# Start camera
cam.start()

# Loop
while True:

    # Read images
    color_image, depth_image = cam.read()

    # Mask ball
    mask = get_mask(color_image, lower_color, upper_color)

    # Get pixel coordinate of ball center
    ball_pixel, radius = get_ball_pixel(mask, minradius)
    
    # Get depth of pixel
    if ball_pixel:
        depth_pixel = cam.get_pixel_depth(depth_image, ball_pixel)
    else:
        depth_pixel = None

    # Transform 2D to 3D camera coordinates
    xcam, ycam, zcam = cam.intrinsic_trans(ball_pixel, depth_pixel, mtx)

    # Transform camera coordinates to world coordinates
    xworld, yworld, zworld = cam.extrinsic_trans(depth_pixel, xcam, ycam, zcam, ext)

    # Show ball and coordinates
    img = viewer.show_ball_pixel(color_image, ball_pixel, depth_pixel, xworld, yworld, zworld, radius)
    img = viewer.draw_axes(img, mtx, dist, rvecs, tvecs, 60)

    # Show image
    viewer.show_image(img, window_name)