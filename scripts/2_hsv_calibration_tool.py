# Imports
import os
import cv2
import numpy
from robot_ball_catcher.Camera import Camera
from robot_ball_catcher.Viewer import Viewer
from robot_ball_catcher.image_processing import *

# Read data from previous calibrations
hsvfile = numpy.load('./data/hsv.npy')

# Window settings
window_name = 'HSV calibration tool'
window_b, window_h = 1500, 1080

# Get camera object
cam = Camera()

# Get Viewer object
viewer = Viewer()

# Set viewer
viewer.set_window(window_name, window_b, window_h)

# Start camera
cam.start()

# Necessary to make sliders
def nothing(*args):
    pass

# Make sliders
cv2.createTrackbar('Hmin', window_name, hsvfile[0], 179, nothing)
cv2.createTrackbar('Hmax', window_name, hsvfile[1], 179, nothing)
cv2.createTrackbar('Smin', window_name, hsvfile[2], 255, nothing)
cv2.createTrackbar('Smax', window_name, hsvfile[3], 255, nothing)
cv2.createTrackbar('Vmin', window_name, hsvfile[4], 255, nothing)
cv2.createTrackbar('Vmax', window_name, hsvfile[5], 255, nothing)
cv2.createTrackbar('save', window_name, 0, 1, nothing)

# Define image formats
HSVmin = numpy.zeros((cam.color_resolution[1], cam.color_resolution[0], 3), numpy.uint8)
HSVmax = numpy.zeros((cam.color_resolution[1], cam.color_resolution[0], 3), numpy.uint8)
HSVgem = numpy.zeros((cam.color_resolution[1], cam.color_resolution[0], 3), numpy.uint8)
white_image = numpy.zeros((cam.color_resolution[1], cam.color_resolution[0], 3), numpy.uint8)

# Initial mask
white_image[:] = [255, 255, 255]

# Loop
while True:

    # Get slider values
    hmin = cv2.getTrackbarPos('Hmin', window_name)
    hmax = cv2.getTrackbarPos('Hmax', window_name)
    smin = cv2.getTrackbarPos('Smin', window_name)
    smax = cv2.getTrackbarPos('Smax', window_name)
    vmin = cv2.getTrackbarPos('Vmin', window_name)
    vmax = cv2.getTrackbarPos('Vmax', window_name)
    save = cv2.getTrackbarPos('save', window_name)

    # Preview
    HSVmin[:] = [hmin, smin, vmin]
    HSVmax[:] = [hmax, smax, vmax]
    HSVgem[:] = [(hmin + hmax) / 2, (smin + smax) / 2, (vmin + vmax) / 2]

    # Convert form HSV to BRG
    BGRmin = cv2.cvtColor(HSVmin, cv2.COLOR_HSV2BGR)
    BGRmax = cv2.cvtColor(HSVmax, cv2.COLOR_HSV2BGR)
    BGRgem = cv2.cvtColor(HSVgem, cv2.COLOR_HSV2BGR)

    # Read images
    color_image, depth_image = cam.read()

    # Define bounds on Hue value
    lower_color = numpy.array([hmin, smin, vmin])
    upper_color = numpy.array([hmax, smax, vmax])

    # Get mask
    mask = get_mask(color_image, lower_color, upper_color)

    # Apply mask to image
    res = cv2.bitwise_and(color_image, color_image, mask=mask)

    # Binary of image
    mask_bgr = cv2.bitwise_and(white_image, white_image, mask=mask)

    # Mount all images
    row1 = numpy.hstack((color_image, mask_bgr, res))
    row2 = numpy.hstack((BGRmin, BGRgem, BGRmax))
    img = numpy.vstack((row1, row2))

    # Show images
    viewer.show_image(img, window_name)

    # Leave loop on save button
    if (save): break

# Save data
hsvarray = numpy.array([hmin, hmax, smin, smax, vmin, vmax])
numpy.save('./data/hsv.npy', hsvarray)