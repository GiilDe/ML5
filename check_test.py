from get_prepared_data import get_unlabeled_data
import pandas as pd
from data_handling import XY_2_X_Y
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

train_XY, test_X = get_unlabeled_data(load=False)
_, test_Y = XY_2_X_Y(pd.read_csv('ElectionsData_Pred_Features for check test.csv'))
train_X, train_Y = XY_2_X_Y(train_XY)

ab = AdaBoostClassifier(RandomForestClassifier(n_estimators=85))
ab.fit(train_X, train_Y)
print(accuracy_score(ab.predict(test_X), test_Y))
