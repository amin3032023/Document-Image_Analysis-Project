#Coener Density merge with Count of black pixels in the sliding window so that we can get at least a little good score 

import cv2
import os
import numpy as np
from PIL import Image
from dtaidistance import dtw_ndim
from skimage.feature import corner_harris, corner_peaks
threshold = 225

def count_black_pixel(image, window_id, window_width):
    count = 0
    for i in range(window_width*window_id, window_width*window_id+window_width):
        col_i = image.T[i]
        for j in col_i:
            if j == 0:
                count += 1
    return count


def compute_corner_density(image):
    if len(image.shape) == 3:  
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image  

    # Ensure the image is of type CV_8UC1 or convert to CV_32FC1
    if gray_image.dtype != np.uint8:
        gray_image = np.uint8(gray_image)
    
    # Ensure the image is of type CV_32FC1
    if gray_image.dtype != np.float32:
        gray_image = np.float32(gray_image)

    corners = cv2.cornerHarris(gray_image, 2, 3, 0.04)
    cv2.normalize(corners, corners, 0, 255, cv2.NORM_MINMAX)
    corners = np.uint8(corners)
    avg_corner_density = np.mean(corners)
    return avg_corner_density


def compute_distance(reference_image, test_image):
    window_width = 4
    x = sharpen_image(reference_image)
    y = sharpen_image(test_image)
    x_nb_window, x_last_window = divmod(x.shape[1], window_width)
    y_nb_window, y_last_window = divmod(y.shape[1], window_width)
    
    serie1 = []
    serie2 = []
    for i in range(x_nb_window):
        if i == x_nb_window-1:
            count_x = count_black_pixel(x, i, x_last_window)
        else:
            count_x = count_black_pixel(x, i, window_width)
        corner_density_x = compute_corner_density(x[:, i * window_width:(i + 1) * window_width])
        serie1.append([count_x, corner_density_x])

    for j in range(y_nb_window):
        if j == x_nb_window - 1:
            count_y = count_black_pixel(y, j, y_last_window)
        else:
            count_y = count_black_pixel(y, j, window_width)
        corner_density_y = compute_corner_density(y[:, j * window_width:(j + 1) * window_width])
        serie2.append([count_y, corner_density_y])

    serie1 = np.array(serie1, dtype=np.double)
    serie2 = np.array(serie2, dtype=np.double)
    return dtw_ndim.distance_fast(serie1, serie2)


def sharpen_image(image):
    sharp_image = np.array(Image.open("../documents/word_images/{}.png".format(image))
                           .convert('L'), dtype=np.double)
    sharp_image[sharp_image > 127] = 255
    sharp_image[sharp_image <= 127] = 0
    return sharp_image


def dtw_score(training_set_id):
    print("Starting DTW score calculation. Minimun 5 Miniyes")
    temp = []
    predicted = []
    all_document_ids = [filename.split('.')[0] for filename in os.listdir('../documents/word_images/')]
    testing_set_id = [doc_id for doc_id in all_document_ids if doc_id not in training_set_id]

    for im_tested in testing_set_id:
        min_distance = None
        for im_reference in training_set_id:
            distance = compute_distance(im_reference, im_tested)
            if min_distance is None or distance < min_distance:
                min_distance = distance
        temp.append(min_distance)

    for i in temp:
        if i < threshold:
            predicted.append(True)
        else:
            predicted.append(False)

    return predicted