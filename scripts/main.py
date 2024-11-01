# Author: Al Amin & Christian Galley
# run 'pip install -r requirements.txt' in a virtual environment
# run 'setup.py' before running this script.

import training_set        # one of our script
import evaluation_metrics  # one of our script
import dtw_script          # one of our script
import Coener_Dencity_Black_pixel_merge


# list also serves us as a blacklist for the test set. (ID to not use!)
word_to_retrieve, training_set_id = training_set.training_set_building()
actual_values = evaluation_metrics.generate_actual_labels(word_to_retrieve, training_set_id)
predicted = dtw_script.dtw_score(training_set_id)
# predicted = Coener_Dencity_Black_pixel_merge.dtw_score(training_set_id)

try:
    evaluation_metrics.prec_rec_curve(actual_values, predicted, word_to_retrieve)
    f1_score = evaluation_metrics.f1(actual_values, predicted)
    print(f"F1 score: {f1_score}")
except NameError:
    print(NameError)
