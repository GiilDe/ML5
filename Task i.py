from Moodel_Chooser import Model_Chooser
from collections import Counter
from get_prepared_data import get_prepared_data
import pandas as pd
import ast
import numpy as np


def which_party_would_win_score(prediction, test_Y):
    prediction_counter = list(Counter(prediction).items())
    prediction_counter.sort(key=lambda x: x[1], reverse=True)
    test_Y_counter = list(Counter(test_Y).items())
    test_Y_counter.sort(key=lambda x: x[1], reverse=True)

    return abs(prediction_counter[0][1] - prediction_counter[1][1] - (test_Y_counter[0][1] - test_Y_counter[1][1]))


train_X, train_Y, validation_X, validation_Y, test_X, test_Y = get_prepared_data()
X_to_split = pd.concat([train_X, validation_X])
Y_to_split = pd.concat([train_Y, validation_Y])

mc = Model_Chooser(score_measurement_for_find_best_params=which_party_would_win_score)
mc.find_classifiers_best_params(X_to_split, Y_to_split)

best_classifier, best_param_list, __ = mc.get_winner(X_to_split, Y_to_split, test_X, test_Y)
best_classifier = mc.create_classifier(best_classifier, ast.literal_eval(best_param_list))
prediction = list(best_classifier.predict(test_X))
print('the predicted winner party is: '.capitalize(), Counter(prediction).most_common(1)[0])

