import numpy as np
import cv2
import matplotlib.image as mpimg
import glob


def __getHLSChannels(img):
    """
    Return H, L and S channels for a supplied image
    """
    hlsImage = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hlsImage[:,:,0]
    l_channel = hlsImage[:,:,1]
    s_channel = hlsImage[:,:,2]
    return l_channel, s_channel, h_channel


def __getGradient(l_channel):
    """
    Calculates the gradient on x, then the absolute value
    and returns the scaled version of the sobel caculation
    based on the Udacity example on the gradients chapter.
    """
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    return np.uint8(255*abs_sobelx/np.max(abs_sobelx))


def combinedThreshold(img, s_thresh=(90, 255), sx_thresh=(20, 100)):
    """
    Transforms an image to a binary thresholded image
    using HLS and getting the image gradient on x.
    The two binary images are stacked into one.
    """
    img = np.copy(img)
    l_channel, s_channel, _ = __getHLSChannels(img)
    scaled_sobel = __getGradient(l_channel)
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return color_binary
