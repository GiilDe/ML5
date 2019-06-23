from Moodel_Chooser import Model_Chooser
from collections import Counter
from get_prepared_data import get_prepared_data, get_unlabeled_data
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
import pickle


mc = Model_Chooser()
mc.find_classifiers_best_params(X_to_split, Y_to_split)

scores_list = mc.get_winner(X_to_split, Y_to_split, test_X, test_Y)
print(scores_list)

abrf = pickle.load(open(''))

real_train_XY, real_test_X = get_unlabeled_data()

predictions = abrf.predict(real_test_X)



