from PIL import Image
import numpy as np
import doxapy
import os
import time

nb_documents = 15


# Check if the directory to store binarized documents exists.
if not os.path.exists('../documents/binarized_images/'):
    os.mkdir("../documents/binarized_images/")


def bin_image(file_in, file_out):
    # locate the file
    grayscale_image = np.array(Image.open("../documents/original_images/"+file_in).convert('L'))
    binary_image = np.empty(grayscale_image.shape, grayscale_image.dtype)

    # algorithm for image binarization
    sauvola = doxapy.Binarization(doxapy.Binarization.Algorithms.SAUVOLA)
    sauvola.initialize(grayscale_image)
    # good parameter for G.W. handwritten documents (window=15, k=0.06)
    sauvola.to_binary(binary_image, {"window": 15, "k": 0.06})

    # Check if directory exist & save images
    if not os.path.exists("../documents/binarized_images/"):
        os.mkdir("../documents/binarized_images/")
        Image.fromarray(binary_image).save("../documents/binarized_images/"+file_out)
    else:
        Image.fromarray(binary_image).save("../documents/binarized_images/"+file_out)


# Get all binarized images if not already done.
# Check if the number of files expected is correct.
if nb_documents != len([name for name in os.listdir('../documents/binarized_images/')
                        if os.path.isfile(os.path.join('../documents/binarized_images/', name))]):
    print('Documents are not binarized yet. Binarizing them now... (time duration: approx. 2 sec)')
    start = time.time()
    # Generate all binarized documents images from G.W.
    for i in range(270, 305):
        # these documents don't exist: we skip this part.
        if 280 <= i < 300:
            pass
        else:
            bin_image("{}.jpg".format(i), "{}b.jpg".format(i))
    print("Binarization process finished in {} seconds".format(round(time.time() - start)))
else:
    print("Binarization is already done.")
