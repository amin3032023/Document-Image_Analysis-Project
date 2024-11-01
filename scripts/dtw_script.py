import os
import numpy as np
from PIL import Image
from dtaidistance import dtw_ndim

threshold = 90


def count_black_pixel(image, window_id, window_width):
    count = 0
    for i in range(window_width*window_id, window_width*window_id+window_width):
        col_i = image.T[i]
        for j in col_i:
            if j == 0:
                count += 1
    return count


def sharpen_image(image):
    sharp_image = np.array(Image.open("../documents/word_images/{}.png".format(image))
                           .convert('L'), dtype=np.double)
    sharp_image[sharp_image > 127] = 255
    sharp_image[sharp_image <= 127] = 0
    return sharp_image


def compute_distance(reference_image, test_image):
    # used to count black pixels
    window_width = 4

    # remove noise and set all pixels to absolute black or white.
    x = sharpen_image(reference_image)
    y = sharpen_image(test_image)

    x_nb_window, x_last_window = divmod(x.shape[1], window_width)
    y_nb_window, y_last_window = divmod(y.shape[1], window_width)

    # start of 1 feature: counting the black pixel in the sliding window.
    # instantiation of time series. Used to store all features.
    serie1 = []
    serie2 = []
    for i in range(x_nb_window):
        if i == x_nb_window-1:
            count_x = count_black_pixel(x, i, x_last_window)
        else:
            count_x = count_black_pixel(x, i, window_width)
        serie1.append([count_x, i])

    for j in range(y_nb_window):
        if j == x_nb_window - 1:
            count_y = count_black_pixel(y, j, y_last_window)
        else:
            count_y = count_black_pixel(y, j, window_width)
        serie2.append([count_y, j])
    # end of feature 1.

    # Start of feature 2
    # TODO


    serie1 = np.array(serie1, dtype=np.double)
    serie2 = np.array(serie2, dtype=np.double)
    return dtw_ndim.distance_fast(serie1, serie2)


def dtw_score(training_set_id):
    print("Starting DTW score calculation. This may take a while... (up to 5 minutes)")
    temp = []
    predicted = []
    all_document_ids = sorted([filename.split('.')[0] for filename in os.listdir('../documents/word_images/')])
    testing_set_id = [doc_id for doc_id in all_document_ids if doc_id not in training_set_id]

    # find minimum distance for each images
    for im_tested in testing_set_id:
        min_distance = None
        # find minimum distance among all targets
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
