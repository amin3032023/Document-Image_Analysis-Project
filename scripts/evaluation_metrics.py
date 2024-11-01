from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import training_set as tr_s
import os
import dtw_script


def generate_actual_labels(word_selected, training_set):
    actual_labels = []
    with open("../documents/ground-truth/transcription.txt", 'r') as file:
        for line in file:
            im_id, word = line.strip().split(' ')  # separate the ID and the transcription

            if im_id in training_set:  # If id is in the training set, we don't add the value to the actual values.
                pass
            else:
                if tr_s.standardize_word(word_selected) == tr_s.standardize_word(word):
                    actual_labels.append(True)
                else:
                    actual_labels.append(False)
    return actual_labels


def prec_rec_curve(actual_labels, predicted_labels, word):
    if not os.path.exists('../documents/results/'):
        os.mkdir("../documents/results/")

    precision, recall, _ = precision_recall_curve(actual_labels, predicted_labels, pos_label=True)
    plt.fill_between(recall, precision)
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.title(f'P/R Curve (word = {word}, threshold = {str(dtw_script.threshold)})')
    # save the results with the chosen threshold. add a number if file already exists.
    if not os.path.exists(f'../documents/results/{word}_th-{str(dtw_script.threshold)}.png'):
        plt.savefig(f'../documents/results/{word}_th-{str(dtw_script.threshold)}.png')
    else:
        i = 1
        while os.path.exists(f'../documents/results/{word}_th-{str(dtw_script.threshold)}({i}).png'):
            i += 1
        plt.savefig(f'../documents/results/{word}_th-{str(dtw_script.threshold)}({i}).png')


def f1(actual_labels, predicted_labels):
    return f1_score(actual_labels, predicted_labels, pos_label=True, average='binary')
