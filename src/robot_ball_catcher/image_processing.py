import cv2

def get_mask(image, mincolor, maxcolor):

    # Remove noise
    blurred = cv2.GaussianBlur(image, (11, 11), 0)

    # Convert to hsv color space
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Get mask
    mask = cv2.inRange(hsv, mincolor, maxcolor)

    # Erode to close gaps
    mask = cv2.erode(mask, None, iterations=2)

    # Dilate to reduce data
    mask = cv2.dilate(mask, None, iterations=2)

    return mask


# Get ball coordinate
def get_ball_pixel(mask, r):

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If ball is present
    if len(contours) > 0:

        # Find contour with largest area
        maxcontour = max(contours, key=cv2.contourArea)

        # Find moments of the largest contour
        moments = cv2.moments(maxcontour)

        # Find ball center with moments
        try:
            center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
        except:
            center = None

        # Find radius of circle
        ((x, y), radius) = cv2.minEnclosingCircle(maxcontour)

        # If radius is big enough, it is the ball
        if radius > r:
            return center, radius
        else:
            return None, None
    else:
        return None, None

# Apply color map to depth image
def color_map(depth_image):
    return cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_HSV)
    







