import numpy as np
import cv2

class Viewer:

    def __init__(self):
        self.window_name = None
        self.window_height = None
        self.window_width = None

    # Draw reference axis
    def draw_axes(self, img, mtx, dist, rvecs, tvecs, length):

        # Frame axis
        axis = np.float32([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, -length]])
        
        # Project 3D points in 2D
        imgpts, _ = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        imgpts = imgpts.astype(int)

        # Line projections
        img = cv2.line(img, tuple(imgpts[3].ravel()), tuple(imgpts[0].ravel()), (0, 0, 255), 5)
        img = cv2.line(img, tuple(imgpts[3].ravel()), tuple(imgpts[1].ravel()), (0, 255, 0), 5)
        img = cv2.line(img, tuple(imgpts[3].ravel()), tuple(imgpts[2].ravel()), (255, 0, 0), 5)
        text_pos = (imgpts[0].ravel() + np.array([3.5, -7])).astype(int)
        cv2.putText(img, 'X', tuple(text_pos), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        text_pos = (imgpts[1].ravel() + np.array([3.5, -7])).astype(int)
        cv2.putText(img, 'Y', tuple(text_pos), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        text_pos = (imgpts[2].ravel() + np.array([3.5, -7])).astype(int)
        cv2.putText(img, 'Z', tuple(text_pos), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        text_pos = (imgpts[3].ravel() + np.array([3.5, -7])).astype(int)

        return img

    # Draw box
    def draw_box(self, img, mtx, dist, rvecs, tvecs, length):

        # Frame axis
        box = np.float32([[0.08, 0.06, 0], [0.12, 0.06, 0], [0.12, 0.10, 0], [0.08, 0.10, 0], [0.08, 0.06, -0.04],
                          [0.12, 0.06, -0.04], [0.12, 0.10, -0.04], [0.08, 0.10, -0.04]]).reshape(-1, 3)
        
        # Project 3D points in 2D
        imgpts, _ = cv2.projectPoints(box, rvecs, tvecs, mtx, dist)
        
        # Line projections
        corner = tuple(imgpts[0].ravel())
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
        img = cv2.line(img, corner, tuple(imgpts[3].ravel()), (255, 0, 0), 5)

        return img

    # Set window
    def set_window(self, windowname, width, height):

        self.window_height = 1080
        self.window_width = 1920

        # Set name
        cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)

        # Set size
        cv2.resizeWindow(windowname, width, height)

    # Show image
    def show_image(self, img, windowname):
        cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowname, 1920, 1080)
        cv2.imshow(windowname, img)
        cv2.waitKey(1)

    # Show ball, pixel, depth and XYZ
    def show_ball_pixel(self, image, pixel, depth, x, y, z, r):

        #Plot ball pixel
        if pixel:
            cv2.circle(image, pixel, 5, (0, 0, 255), -1)
            cv2.circle(image, pixel, int(r), (255, 0, 0), 5)

        center_as_string = ''.join(str(pixel))

        # Too close to detect depth
        if depth == 0:
            depth_as_string = 'TO CLOSE'
        else:
            depth_as_string = str(depth)

        # Project pixel and depth values
        cv2.putText(image, "pixel: " + center_as_string, (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "depth in mm: " + depth_as_string, (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Project 3D cooridnates
        if depth:
            if x or y or z:
                cv2.putText(image, "X: " + str(int(round(x))) + ' mm', (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, "Y: " + str(int(round(y))) + ' mm', (10, 600), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, "Z: " + str(int(round(z))) + ' mm', (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        
        return image