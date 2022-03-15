# Imports
import numpy
from robot_ball_catcher.Camera import Camera
from robot_ball_catcher.Viewer import Viewer
from robot_ball_catcher.image_processing import *

# Window settings
window_name = 'Camera viewer'
window_b, window_h = 960, 1080

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

    # Retrieve images
    color_image, depth_image = cam.read()

    # Apply color map to depth image
    depth_colormap = color_map(depth_image)

    # Mount images
    images = numpy.vstack((color_image, depth_colormap))
    
    # Show images
    viewer.show_image(images, window_name)
