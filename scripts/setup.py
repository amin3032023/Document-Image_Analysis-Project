# Binarize all documents if not done already.
with open('binarization.py') as file:
    exec(file.read())

# Crop all words into single image if not done already.
with open('word_image_separation.py') as file:
    exec(file.read())

# PREPROCESSING IS OVER.
