from collections import Counter
import numpy as np


def division_of_votes_score(prediction, test_Y):
    prediction_counter = list(Counter(prediction).items())
    test_Y_counter = list(Counter(test_Y).items())
    prediction_counter.sort(key=lambda x: x[0])
    test_Y_counter.sort(key=lambda x: x[0])

    prediction_counter_counts = np.array([x[1] for x in prediction_counter])
    test_Y_counter_counts = np.array([x[1] for x in test_Y_counter])
    return np.linalg.norm(prediction_counter_counts - test_Y_counter_counts, ord=1)

