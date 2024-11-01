
training_set_size = 0.6  # setting the size of training set to x% of total occurrence of the word. (rounded below)
training_set_path = "../documents/training_dataset/"


# remove all signs.
def standardize_word(word):
    return (word.replace("-", "").replace("s_pt", "").replace("s_cm", "").replace("s_mi", "")
            .replace("s_sq", "").replace("_qt", " ").replace("_qo", " ").replace("s_", ""))


# Function to find the image ID corresponding to the input word
def training_set_building():
    word_to_retrieve = input("Please enter the word to search for: ")
    id_list = []
    with open("../documents/ground-truth/transcription.txt", 'r') as file:
        for line in file:
            im_id, word = line.strip().split(' ')  # separate the ID and the transcription
            if standardize_word(word_to_retrieve) == standardize_word(word):
                id_list.append(im_id)

    # At this point we have retrieve all images similar to the input. Now we build the training set.
    # Covers all specific cases
    if len(id_list) == 0:
        print("Word '{}' does not appear in documents.".format(word_to_retrieve))
        return word_to_retrieve, []
    elif len(id_list) == 1:
        print("Word '{}' only appears once in documents.".format(word_to_retrieve))
        return word_to_retrieve, []
    elif len(id_list) == 2:
        tr_set = id_list[0]
        return word_to_retrieve, tr_set
    else:
        tr_set = id_list[:int(len(id_list)*training_set_size)]
        return word_to_retrieve, tr_set
