# Imports
import numpy
from robot_ball_catcher.Camera import Camera
from robot_ball_catcher.Viewer import Viewer
from robot_ball_catcher.image_processing import *

# Get HSV calibration params
hsvfile = numpy.load('./data/hsv.npy')

# Define upper and lower Hue value
lower_color = numpy.array([hsvfile[0], hsvfile[2], hsvfile[4]])
upper_color = numpy.array([hsvfile[1], hsvfile[3], hsvfile[5]])

# Window settings
window_name = 'UVZ ball detection'
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
    
    # Show ball and pixel and depth values
    img = viewer.show_ball_pixel(color_image, ball_pixel, depth_pixel, None, None, None, radius)
    
    # Show image
    viewer.show_image(img, window_name)