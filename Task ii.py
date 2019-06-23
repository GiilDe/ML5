from collections import Counter
import numpy as np
from Moodel_Chooser import Model_Chooser
from get_prepared_data import get_unlabeled_data
import pandas as pd
import pickle as pk
import data_handling
from sklearn.metrics import accuracy_score
from party_num_to_name import parties_dict

train, test = get_unlabeled_data(load=True)
train_X, train_Y, validation_X, validation_Y, test_X, test_Y = data_handling.split_data(train)
X_to_split = pd.concat([train_X, validation_X])
Y_to_split = pd.concat([train_Y, validation_Y])
all_data_X, all_data_Y = data_handling.XY_2_X_Y(train)


def get_party_hist(Y):
    hist = np.unique(Y, return_counts=True)
    res = np.zeros(13)
    for label, size in zip(hist[0], hist[1]):
        res[label] = size
    return res


def division_of_votes_score(Y_hat, Y):
    Y_hat_hist = get_party_hist(Y_hat)
    Y_hist = get_party_hist(Y)
    return 1/np.linalg.norm(Y_hat_hist - Y_hist, ord=1)


mc = Model_Chooser(score_measurement_for_find_best_params=division_of_votes_score)
mc.find_classifiers_best_params(X_to_split, Y_to_split)
model_name, params, _ = mc.get_winner(X_to_split, Y_to_split, test_X, test_Y, score_measure_func=division_of_votes_score)
model = mc.create_classifier(model_name, params)
model.fit(all_data_X, all_data_Y)
pk.dump(model, open("model for division of voters", 'wb'))
Y_hat = model.predict(test)
Y_hat_train = model.predict(all_data_X)
division = get_party_hist(Y_hat)
division_list = []
for i in range(13):
    division_list.append((parties_dict[i], division[i]))

division_list.sort(key=lambda x: x[1])
print(division_list)
