#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML
# matplotlib inline

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=8):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    bottom_y = img.shape[0]
    top_y = int(bottom_y /1.6)
    left_lines = []
    right_lines = []

    for line in lines:
        for x1,y1,x2,y2 in line:
            # b = y - xm
            slope = (y2-y1)/(x2-x1)
            if np.absolute(slope) == np.inf or np.absolute(slope) < 0.5:
                pass
            else:
                if slope < 0:
                    left_lines.append((x1,y1))
                    left_lines.append((x2,y2))
                else:
                    right_lines.append((x1,y1))
                    right_lines.append((x2,y2))

    if len(left_lines) > 0 and len(right_lines) > 0  :
        [left_vx,left_vy,left_x,left_y] = cv2.fitLine(np.array(left_lines, dtype=np.int32), cv2.DIST_L2,0,0.01,0.01)
        [right_vx,right_vy,right_x,right_y] = cv2.fitLine(np.array(right_lines, dtype=np.int32),cv2.DIST_L2,0,0.01,0.01)

        left_slope = left_vy / left_vx
        left_b = left_y - (left_slope*left_x)
        right_slope = right_vy / right_vx
        right_b = right_y - (right_slope*right_x)

        # Slopes and intercepts ready to use
        left_top_x = (top_y - left_b) / left_slope
        left_bottom_x = (bottom_y - left_b) / left_slope

        right_top_x = (top_y - right_b) / right_slope
        right_bottom_x = (bottom_y - right_b) / right_slope

        # Finally Draw the full lines
        cv2.line(img, (left_bottom_x, bottom_y), (left_top_x, top_y), color, thickness)
        cv2.line(img, (right_bottom_x, bottom_y), (right_top_x, top_y), color, thickness)



def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def identify_lines_for(image):
    gray = grayscale(image)

    kernel_size = 5
    blur_gray = gaussian_blur(image, kernel_size)

    low_threshold = 50
    high_threshold = 150
    edges = canny(gray, low_threshold, high_threshold)

    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(510, 306), (510, 306), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40 #minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments
    lines = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    lines_edges = weighted_img(lines, image)

    return lines_edges

def process_image(image):
    print(type(image))
    return identify_lines_for(image)

# --------- Making the pipeline for Line detection

# Run pipeline on images
# folder = "test_images/"
# all_images = os.listdir(folder)
# for image_path in all_images:
#     #reading in an image
#     image = mpimg.imread(folder + image_path)
#     image_with_lines = identify_lines_for(image)
#     # Save my image
#     from PIL import Image
#     im = Image.fromarray(image_with_lines)
#
#     # new_file_name = image_path.replace('.jpg', '_edwin_test.jpg')
#     new_file_name = folder + image_path # This will overwrite the existing images.
#     im.save(new_file_name)

# VideoFileClip
white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
# --------
