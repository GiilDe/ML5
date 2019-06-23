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

    predicted_winner = prediction_counter[0][0]
    real_winner = test_Y_counter[0][0]

    if real_winner != predicted_winner:
        return float('-inf')

    predicted_winner_votes = prediction_counter[0][1]
    real_winner_votes = test_Y_counter[0][1]

    predicted_second_votes = prediction_counter[1][1]
    real_second_votes = test_Y_counter[1][1]

    return min((predicted_winner_votes - predicted_second_votes) - (real_winner_votes - real_second_votes), 0)


train_X, train_Y, validation_X, validation_Y, test_X, test_Y = get_prepared_data()
X_to_split = pd.concat([train_X, validation_X])
Y_to_split = pd.concat([train_Y, validation_Y])

mc = Model_Chooser(score_measurement_for_find_best_params=which_party_would_win_score)
mc.find_classifiers_best_params(X_to_split, Y_to_split)

best_classifier, best_param_list, __ = mc.get_winner(X_to_split, Y_to_split, test_X, test_Y, which_party_would_win_score)
print(best_classifier)
best_classifier = mc.create_classifier(best_classifier, best_param_list)
print('best_classifier: ', best_classifier)
print('best_param_list: ', best_param_list)
best_classifier.fit(X_to_split, Y_to_split)
prediction = list(best_classifier.predict(test_X))
print('the predicted winner party is: '.capitalize(), Counter(prediction).most_common(1)[0])
