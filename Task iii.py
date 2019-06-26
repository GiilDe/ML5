from Moodel_Chooser import Model_Chooser
from collections import Counter
from get_prepared_data import get_prepared_data, get_unlabeled_data
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
import pickle as pk
import data_handling
from party_num_to_name import parties_dict


model = RandomForestClassifier(n_estimators=85)
ad = AdaBoostClassifier(model)

train, test = get_unlabeled_data(load=True)
train_X, train_Y = data_handling.XY_2_X_Y(train)
ad.fit(train_X, train_Y)
test_Y_hat = ad.predict(test)
test_Y_hat_names = []
for i in range(len(test_Y_hat)):
    party_num = test_Y_hat[i]
    party_name = parties_dict[party_num]
    test_Y_hat_names.append(party_name)

res = pd.DataFrame()
res['predictions'] = pd.Series(test_Y_hat_names)
    
res['IdentityCard_Num'] = pd.Series(i for i in range(len(test)))
res.to_csv('predictions.csv', index=False)

