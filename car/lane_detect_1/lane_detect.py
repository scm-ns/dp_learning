import matplotlib.pyplot as plt
import matplotlib.images as implt
import numpy as np
import cv2
from math import pi

image = implt.imread(im1)
print("image type : " , type(image) , "dims : " , image.shape)

def gray_scale(img):
    cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

def gray_scale_hsv(img):
    hsv = cv2.cvtColor(img , cv2.COLOR_BGR2HSV)

def canny(img , low , high):
    return cv2.Canny(img , low , high)

def gaussian_noise(img , kernel_size):
    return cv2.GaussianBlur(img , (kernel_size , kernel_size) , 0)

def roi(img , vertx):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        mask_color = (255,) * channel_count
    else:
        mask_color = 255

    cv2.fillPoly(mask , vertx , mark_color)
    masked_img = cv2.bitwise_and(img , mask)
    return masked_img

def hough_lines(img , rho , theta , threshold , min_line_len , max_line_gap):
    lines = cv2.HoughLines(img , rho , theta , threshold , np.array([]) , minLineLenght = min_line_len , maxLineGap = max_line_gap)
    return lines


def weights_img(img1 , alpha = 0.8 , im2 , beta = 1 , gamma = 0 ):
    return cv2.addWeighted(im1, alpha , im2 , beta , gamma)


def draw_lines(lines , imshape , color= [255 , 0 , 0 ] , thickness = 10):
    if len(imshape) < 3:
        imshape  = (imshape[0] , imshape[1] , 3)

    line_img = np.zeros(imshape , dtype = np.uint8)
    for line in lines:
        for x1 , y1 , x2 , y1 in line:
            cv2.line(line_img , (x1 , y1) , (x2 , y2) , color, thickness)
    
    return line_img



from moviepy.editor import VideoFileClip

def detect_lanes(lines , imshape , angle_min_mag = 25 * p1/180 , angle_max_mag = 40*pi/180 ,
        rho_min_diag = 0.1 , rho_max_diag = 0.60 , cache_wt = 0.90, last_lane = None):
    lane_marker_x = [[] , []]
    lane_marker_y = [[] . []]

    if last_lane is not None:
        last_apex_pt = np.array([last_lanes[0][0][2] , last_lanes[0][0][3])
        last_left_pt = last_lanes[0][0][1]
        last_right_pt = last_lanes[1][0][1]

    diag_len = math.sqrt(imshape[0]**2 + imshape[1]**2)
    for line in lines:
        for x1 , y1 , x2 , y2 in line:
            theta = math.atan2(y1 - y2 , x2 - x1)
            rho = ((x1 + x2 ) * math.cos(theta) _+ (y1 + y2)* math.sin(theta)) / 2


            if(abs(theta) >= angle_min_mag and abs(theta) <= angle_max_mag
                and rho >= rho_min_diag * diag_len and rho <= rho_max_diag * diag_len):
                if theta > 0 :
                    lane_idx = 0 
                else:
                    lane_idx = 1
                lane_marker_x[lane_idx].append(x1)
                lane_marker_x[lane_idx].append(x2)
                lane_marker_y[lane_idx].append(y1)
                lane_marker_y[lane_idx].append(y2)


    if( len(lane_markers_x[0]) == 0 or 
        len(lane_markers_y[1]) == 0 or
        len(lane_marker_y[0]) == 0 or 
        len(lane_marker_y[1]) == 0):
        
        if last_lane is not None:
            apex_pt = last_apex_pt
            left_pt = last_left_pt
            right_pt = last_right_pt
        else:
            return None

    else:
        p_left = np.



































