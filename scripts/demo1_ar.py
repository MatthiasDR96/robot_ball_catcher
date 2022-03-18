# Imports
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import cv2
import imutils
import numpy as np
from collections import deque
import numpy
from matplotlib.animation import FuncAnimation
from robot_ball_catcher.Camera import Camera
from robot_ball_catcher.Viewer import Viewer
from robot_ball_catcher.image_processing import *


def plot_coordinates(d, x, y, z):

    # Pop previous data
    X.popleft()
    Y.popleft()
    Z.popleft()

    # Append new data
    if d:
        X.append(x)
        Y.append(y)
        Z.append(z)
    else:
        X.append(0)
        Y.append(0)
        Z.append(0)

    # Clear axis
    ax1.cla()
    ax2.cla()
    ax3.cla()

    # Set canvas data
    ax1.set_title('X-coordinate (mm)')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('X-coordinate (mm)')
    ax1.set_ylim(-300, 300)
    ax2.set_title('Y-coordinate (mm)')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Y-coordinate (mm)')
    ax2.set_ylim(-300, 300)
    ax3.set_title('Z-coordinate (mm)')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Z-coordinate (mm)')
    ax3.set_ylim(-300, 300)

    # Plot axis
    ax1.plot(X)
    ax2.plot(Y)
    ax3.plot(Z)

    # Tight layout
    fig.tight_layout()

# Update animation function
def update(i):

    # Read images
    color_image, depth_image = cam.read()

    # Gaussian blur
    blurred_image = cv2.GaussianBlur(color_image, (7, 7), 0)

    # Canny edge detection
    canny_image = cv2.Canny(blurred_image, 50, 200)
    canny_image = cv2.bitwise_and(color_image, color_image, mask=canny_image)

    # Mask ball
    mask_image = get_mask(color_image, lower_color, upper_color)

    # Apply mask to image
    res = cv2.bitwise_and(color_image, color_image, mask=mask_image)

    # Get pixel coordinate of ball center
    ball_pixel, radius = get_ball_pixel(mask_image, 30)

    # Contour detection
    contours = cv2.findContours(mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours_image = cv2.drawContours(color_image.copy(), contours, -1, (0, 255, 0), 3)

    # Compute the bounding box of the contour
    if not len(contours) == 0:
        c = max(contours, key = cv2.contourArea)
        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box_image = cv2.drawContours(color_image.copy(), [box.astype("int")], -1, (0, 255, 0), 2)
        center = np.mean(box, axis=0).astype(int)
        cv2.circle(box_image, tuple(center), 3, [255, 0, 0], thickness=3)
    else:
        box_image = contours_image
    
    # Exctrinsic calibration
    ret, corners, rvecs, tvecs, ext = cam.extrinsic_calibration(color_image)

     # Create image 2
    final_img_2 = color_image.copy()
    if ret:
        final_img_2 = cv2.drawChessboardCorners(final_img_2, (cam.b, cam.h), corners, ret)
        final_img_2 = viewer.draw_axes(final_img_2, cam.mtx, cam.dist, rvecs, tvecs, 52.2)
        final_img_2 = viewer.draw_cube(final_img_2, cam.mtx, cam.dist, rvecs, tvecs, 52.2, 139.2, 0, 52.2)

    # If ball found
    if ball_pixel and ret:

        # Get pixel depth
        depth_pixel = cam.get_pixel_depth(depth_image, ball_pixel)

        # Transform 2D to 3D camera coordinates
        xcam, ycam, zcam = cam.intrinsic_trans(ball_pixel, depth_pixel, mtx)

        # Transform camera coordinates to world coordinates
        yworld, xworld, zworld = cam.extrinsic_trans(depth_pixel, xcam, ycam, zcam, ext)

        # Show ball and coordinates
        final_img_2 = viewer.show_ball_pixel(final_img_2, ball_pixel, depth_pixel, xworld, yworld, zworld, radius)

        # Plot coordinates
        plot_coordinates(depth_pixel, xworld, yworld, zworld)

    # Create image 1
    row1 = numpy.hstack((color_image, blurred_image, res))
    row2 = numpy.hstack((canny_image, contours_image, box_image))
    final_img_1 = numpy.vstack((row1, row2))

    # Show images
    viewer.show_image(final_img_1, 'Image processing')
    viewer.show_image(final_img_2, 'Augmented reality')

if __name__ == '__main__':

    # Get camera object
    cam = Camera()

    # Get Viewer object
    viewer = Viewer()

    # Start camera
    cam.start()

    # Initial mask
    HSVmin = numpy.zeros((cam.color_resolution[1], cam.color_resolution[0], 3), numpy.uint8)
    HSVmax = numpy.zeros((cam.color_resolution[1], cam.color_resolution[0], 3), numpy.uint8)
    HSVgem = numpy.zeros((cam.color_resolution[1], cam.color_resolution[0], 3), numpy.uint8)
    white_image = numpy.zeros((cam.color_resolution[1], cam.color_resolution[0], 3), numpy.uint8)
    white_image[:] = [255, 255, 255]

    # Get HSV calibration params path
    hsvfile = numpy.load('c:/Users/Admin/Documents/robot_ball_catcher/data/hsv.npy')

    # Get HSV calibration params
    lower_color = numpy.array([hsvfile[0], hsvfile[2], hsvfile[4]])
    upper_color = numpy.array([hsvfile[1], hsvfile[3], hsvfile[5]])

    # Get camera calibration params
    mtx = numpy.load('c:/Users/Admin/Documents/robot_ball_catcher/data/intrinsics.npy')
    dist = numpy.load('c:/Users/Admin/Documents/robot_ball_catcher/data/distortion.npy')
    ext = numpy.load('c:/Users/Admin/Documents/robot_ball_catcher/data/extrinsics.npy')
    rvecs = numpy.load('c:/Users/Admin/Documents/robot_ball_catcher/data/extrinsic_rvecs.npy')
    tvecs = numpy.load('c:/Users/Admin/Documents/robot_ball_catcher/data/extrinsic_tvecs.npy')

    # Set canvas
    fig = plt.figure()
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)

    # Set lists
    X = deque(numpy.zeros(100))
    Y = deque(numpy.zeros(100))
    Z = deque(numpy.zeros(100))

    # Animate
    ani = FuncAnimation(fig, update, interval=1)

    # Show
    plt.show()