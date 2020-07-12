import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from line import Line
from moviepy.editor import VideoFileClip

# global variables , tweak this to change the end result of the image/video
left_line = Line()
right_line = Line()
objpoints, imgpoints = [[] for x in range(0,2)] # calibration vaues
# image global vars
width = 1280    # hardcode image size to avoid over calculate by video frames
height = 720
horizon = (height/2)+100
mid_point = (width/2) + 20
multiplier = 50
left_margin = 190 - multiplier
right_margin = width - 160  + multiplier
finish = height
mid_point_left = 63 + 10
mid_point_right = 65 + 10
margin = 100
#  # straight lines
#  left_margin = 190
#  height, width, _ = img.shape
#  horizon = (height/2)+100
#  mid_point = (width/2)
#  mid_point_left = 63
#  mid_point_right = 65
#  right_margin = width-160
#  finish = height - 30
#  margin = 100

# filter global vars
s_thresh_low = 130
s_thresh_high = 250
sx_thresh_low = 40
sx_thresh_high = 175
# extra filters added to get a cleaner ourput lines
low_threshold = 0
high_threshold = 255

# output global vars
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# text values definition
font                   = cv2.FONT_HERSHEY_SIMPLEX
firstTopLeftCornerOfText = (10,50)
secondTopLeftCornerOfText = (10,80)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

# Applies the Grayscale transform
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# plot  more than one image
def plot_multiple(figures, nrows=1, ncols=1):
    h, w = 10, 10        # for raster image
    figsize = [6, 8]     # figure size, inches
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for i, axi in enumerate(ax.flat):
        axi.imshow(figures[i])
        # get indices of row/column
        rowid = i // ncols
        colid = i % ncols
        # write row/col indices as axes' title for identification
        axi.set_title("Row:"+str(rowid)+", Col:"+str(colid))
    plt.tight_layout(True)
    plt.show()

# get camera calibration using chessboard images
def get_calibration(images):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # chessboard corners
    nx = 9
    ny = 6

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # DEBUG: uncomment this for debugging
            # Draw and display the corners
            # cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
    return objpoints, imgpoints

# calibratee image using the output points from the chessboard calibration
def calibrate_image(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

# get image lines using gradient thresholds and Sobel
def get_gradient(img, s_thresh=(130,250), sx_thresh=(10, 175), low_threshold=100, high_threshold=180):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    gray = grayscale(color_binary)

    # extra filter to smoth the output image / not using it right now
    gray[(s_channel <= low_threshold) & (s_channel >= high_threshold)] = 0
    return gray

# get a region from an image
def region_of_interest(img):
    pts = np.array([[left_margin,finish],[mid_point - mid_point_left, horizon],[mid_point + mid_point_right, horizon], [right_margin, finish], [0,finish]])
    vertices = np.array([pts], dtype=np.int32)

    # defining a blank mask to start with
    mask = np.zeros_like(img)
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# add solid lines for the reg
def debug_reg_lines(img):
    pts = np.array([[left_margin,finish],[mid_point - mid_point_left, horizon],[mid_point + mid_point_right, horizon], [right_margin, finish], [0,finish]])
    vertices = np.array([pts], dtype=np.int32)
    cv2.polylines(img,[vertices],True,(0,255,255))

# get top view / apply perspective to an image
def get_top_view(img):
    warped = np.copy(img)
    src = np.float32([[mid_point - mid_point_left, horizon],[mid_point + mid_point_right, horizon], [right_margin, finish], [left_margin,finish]])
    dst = np.float32([[margin,margin],[width-margin, margin],[width-margin, height-margin], [margin, height-margin]])
    M = cv2.getPerspectiveTransform(src, dst)
    img_size = (warped.shape[1], warped.shape[0])
    warped = cv2.warpPerspective(warped, M, img_size, flags=cv2.INTER_LINEAR)

    return warped

# from top view to front viewa / probably this could be done getting the inverse of the top_view matrix
def get_original_view(img):
    warped = np.copy(img)
    src = np.float32([[mid_point - mid_point_left, horizon],[mid_point + mid_point_right, horizon], [right_margin, finish], [left_margin,finish]])
    dst = np.float32([[margin,margin],[width-margin, margin],[width-margin, height-margin], [margin, height-margin]])
    M = cv2.getPerspectiveTransform(dst, src)
    img_size = (warped.shape[1], warped.shape[0])
    warped = cv2.warpPerspective(warped, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

def get_histogram_data(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    return out_img, leftx_base, rightx_base


# using a top view image finde the lanes
def find_lane_pixels(binary_warped, leftx_base, rightx_base):
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # get the first/start of the lines
    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # DEBUG: uncomment this for debugging
        # Draw the windows on the visualization image
        #  cv2.rectangle(binary_warped,(win_xleft_low,win_y_low),
        #  (win_xleft_high,win_y_high),(0,255,0), 2)
        #  cv2.rectangle(binary_warped,(win_xright_low,win_y_low),
        #  (win_xright_high,win_y_high),(0,255,0), 2)


        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window
        # (`right` or `leftx_current`) on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


# find the lines inside the windows / get the curvature + plot the lines
def fit_polynomial(leftx, lefty, rightx, righty, binary_warped):

    # Fit a second order polynomial to each using `np.polyfit`#
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    binary_warped[lefty, leftx] = [255, 0, 0]
    binary_warped[righty, rightx] = [0, 0, 255]

    # get green area between the two lines
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(binary_warped, np.int_([pts]), (0,255, 0))

    # DEBUG : uncomment for debugging
    # Plots the left and right polynomials on the lane lines
    #  plt.plot(left_fitx, ploty, color='yellow')
    #  plt.plot(right_fitx, ploty, color='yellow')
    #  cv2.polylines(binary_warped, np.array([np.column_stack([right_fitx, ploty])], dtype=np.int32),False, [255,0,0], 10)
    #  cv2.polylines(binary_warped, np.array([np.column_stack([left_fitx, ploty])], dtype=np.int32),False, [0,0,255], 10)

    # measure the curvature px per meters
    left_fit_cr = np.polyfit(lefty*ym_per_pix, xm_per_pix*leftx, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, xm_per_pix*rightx, 2)
    left_curverad, right_curverad = measure_curvature_real(ploty, left_fit_cr, right_fit_cr)

    return left_curverad, right_curverad

def search_around_poly(binary_warped, left_line, right_line):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_line.current_fit[0]*(nonzeroy**2) + left_line.current_fit[1]*nonzeroy +
                    left_line.current_fit[2] - margin)) & (nonzerox < (left_line.current_fit[0]*(nonzeroy**2) +
                    left_line.current_fit[1]*nonzeroy + left_line.current_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_line.current_fit[0]*(nonzeroy**2) + right_line.current_fit[1]*nonzeroy +
                    right_line.current_fit[2] - margin)) & (nonzerox < (right_line.current_fit[0]*(nonzeroy**2) +
                    right_line.current_fit[1]*nonzeroy + right_line.current_fit[2] + margin)))
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

# Calculates the curvature of polynomial functions in meters
def measure_curvature_real(ploty, left_fit_cr, right_fit_cr):
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Implement the calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_curverad, right_curverad

# plot lines and text on top of the final image
def get_final_image(original, out_img, car_position):
    projected_lines = get_original_view(out_img)
    output = cv2.addWeighted(original, 1, projected_lines, 0.3, 0)

    cv2.putText(output, f"Radius of curvature: {round(np.mean([left_line.get_average_curvature(), right_line.get_average_curvature()]),3)}m",
    #  cv2.putText(output, f"Radius of curvature: {round(left_line.get_average_curvature(),3)} {round(right_line.get_average_curvature(),3)}m",
            firstTopLeftCornerOfText, font, fontScale, fontColor, lineType)
    if car_position > 0:
        car_position = f"{round(abs(car_position) * xm_per_pix,3)}m right"
    elif car_position < 0:
        car_position = f"{round(abs(car_position) * xm_per_pix,3)}m left"
    else:
        car_position = f"{round(abs(car_position) * xm_per_pix,3)}m"

    cv2.putText(output, f"Vehicle is {car_position} of center",
            secondTopLeftCornerOfText, font, fontScale, fontColor, lineType)

    return output

# process each frame/image
def process_image(img):
    # calibrate the camera
    calibrated = calibrate_image(img, objpoints, imgpoints)
    # get the gradient image
    gradient = get_gradient(calibrated, s_thresh=(s_thresh_low, s_thresh_high), sx_thresh=(sx_thresh_low, sx_thresh_high), low_threshold=low_threshold, high_threshold=high_threshold)
    # DEBUG: uncomment to debug regions
    #  debug_reg_lines(gradient)
    # trim the image to work only on the region of interest
    reg = region_of_interest(gradient)
    # get top view proyection
    top_view = get_top_view(reg)

    #  plot_multiple([top_view, gradient], nrows=1, ncols=2)
    #  plt.imshow(top_view)
    #  plt.show()
    if (left_line.detected and right_line.detected): # mostly for video
        binary_warped = np.dstack((top_view, top_view, top_view))
        leftx, lefty, rightx, righty = search_around_poly(binary_warped, left_line, right_line)
        left_curverad, right_curverad = fit_polynomial(leftx, lefty, rightx, righty, binary_warped)
    else: # single iteration = for image
        binary_warped, leftx_base, rightx_base = get_histogram_data(top_view)
        leftx, lefty, rightx, righty = find_lane_pixels(binary_warped, leftx_base, rightx_base)
        left_curverad, right_curverad = fit_polynomial(leftx, lefty, rightx, righty, binary_warped)
    # update lines
    left_line.update(leftx, lefty, left_curverad, abs(top_view.shape[1]/2 - np.average(leftx)))
    right_line.update(rightx, righty, right_curverad, abs(top_view.shape[1]/2 - np.average(rightx)))
    # calculate car position from the lines
    car_position = right_line.line_base_pos - left_line.line_base_pos

    # plot the results on the image
    final_image = get_final_image(calibrated, binary_warped, car_position)

    return final_image

if __name__ == '__main__':
    # get camera calibration
    objpoints, imgpoints = get_calibration(glob.glob('camera_cal/calibration*.jpg'))

    # process images
    #  images = sorted(glob.glob('test_images/test*.jpg'))
    #  #  images = sorted(glob.glob('test_images/straight_lines*.jpg'))
    #  img_name = images[5]
    #  print(img_name)
    #  output_img = process_image(cv2.imread(img_name))
    #  #  cv2.imwrite(f'output_images/{img_name.split("/")[-1]}', output_img)
    #  plt.imshow(output_img)
    #  plt.show()

    # process video
    white_output = 'output_videos/project_video_2.mp4'
    clip1 = VideoFileClip("project_video.mp4", audio=False)
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(white_output, audio=False, preset='ultrafast')
