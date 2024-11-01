from PIL import Image, ImageDraw
import numpy as np
import xml.etree.ElementTree as ET
import os
import time

path_images = "../documents/binarized_images/"
path_shape_of_images = "../documents/word_images/"
path_locations = "../documents/ground-truth/locations/"
nb_words = 3726


# Check if the directory to store word images exists.
if not os.path.exists('../documents/word_images/'):
    os.mkdir("../documents/word_images/")


# fct to crop words based on shapes calculated afterward.
def crop_words(list_doc_file, file_nb, word, all_coord, all_box, id_word):
    im = Image.open(path_images + list_doc_file[file_nb]).convert("RGBA")
    imArray = np.asarray(im)
    polygon = all_coord[word]
    maskIm = Image.new('L', (imArray.shape[1], imArray.shape[0]), 0)
    ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
    mask = np.array(maskIm)
    newImArray = np.empty(imArray.shape, dtype='uint8')
    newImArray[:, :, :3] = imArray[:, :, :3]
    # transparency (4th column)
    newImArray[:, :, 3] = mask * 255
    # back to Image from numpy

    newIm = Image.fromarray(newImArray, "RGBA")
    im_crop = newIm.crop((all_box[word][0], all_box[word][1], all_box[word][2], all_box[word][3]))

    # Create a blank image with the dimensions of the bounding box
    blank_image = Image.new('RGBA', (all_box[word][2] - all_box[word][0], all_box[word][3] - all_box[word][1]),
                            (255, 255, 255, 255))

    # Paste the cropped word image onto the blank image
    paste_position = (0, 0)
    blank_image.paste(im_crop, paste_position, im_crop)

    # Save the result
    if not os.path.exists(path_shape_of_images):
        os.makedirs(path_shape_of_images)
    blank_image.save(path_shape_of_images + '{}.png'.format(id_word[word]))



# Crop all words of all documents if not already done. (time duration: approx. 7min)
# Check if the number of files expected is correct.
if nb_words != len([name for name in os.listdir('../documents/word_images/')
                    if os.path.isfile(os.path.join('../documents/word_images/', name))]):
    print('Words are not cropped yet. Cropping them now... (time duration: approx. 7 min)')
    start = time.time()

    list_path_svg = []
    list_path_doc = []
    for i in range(270, 305):
        if 280 <= i < 300:
            pass
        else:
            list_path_svg.append("{}{}.svg".format(path_locations, str(i)))
            list_path_doc.append("{}b.jpg".format(str(i)))

    # Loop through all SVG files
    for ind, page in enumerate(list_path_svg):

        # Select one page
        tree = ET.parse(list_path_svg[ind])
        all_coordinates_of_words = []
        all_word_box = []
        words_id = []

        # Retrieve shapes of word boxes
        for elem in tree.findall(".//*[@id]"):
            x_ax, y_ax = 10000, 10000  # Initialize the smallest x and y coordinates for each word
            width, height = 0, 0  # Initialize the width and height of each word
            word_coord = []
            list_of_coord = elem.get('d').split()
            for index, element in enumerate(list_of_coord):
                if element == 'M':
                    point = (float(list_of_coord[index + 1]), float(list_of_coord[index + 2]))
                    word_coord.append(point)
                elif element == 'L':
                    point = (float(list_of_coord[index + 1]), float(list_of_coord[index + 2]))
                    word_coord.append(point)
                    # Find the smallest rectangle to crop the image
                    if min(float(list_of_coord[index - 2]), float(list_of_coord[index + 1])) < x_ax:
                        x_ax = min(round(float(list_of_coord[index - 2])), round(float(list_of_coord[index + 1])))
                    if min(float(list_of_coord[index - 1]), float(list_of_coord[index + 2])) < y_ax:
                        y_ax = min(round(float(list_of_coord[index - 1])), round(float(list_of_coord[index + 2])))
                    if max(float(list_of_coord[index - 2]), float(list_of_coord[index + 1])) > width:
                        width = max(round(float(list_of_coord[index - 2])), round(float(list_of_coord[index + 1])))
                    if max(float(list_of_coord[index - 1]), float(list_of_coord[index + 2])) > height:
                        height = max(round(float(list_of_coord[index - 1])), round(float(list_of_coord[index + 2])))
            # Store the shape of the word and retrieve its ID
            all_coordinates_of_words.append(word_coord)
            all_word_box.append([x_ax, y_ax, width, height])
            words_id.append(elem.get('id'))

        # Crop images for all words in the page
        for i in range(len(all_coordinates_of_words)):
            crop_words(list_path_doc, ind, i, all_coordinates_of_words, all_word_box, words_id)
        print("All words of document #{} have been cropped".format(ind+1))

    print("Cropping process finished in {} seconds".format(round(time.time() - start)))
else:
    print("Image Cropping is already done.")



