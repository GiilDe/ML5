from get_prepared_data import get_unlabeled_data
import pandas as pd
from data_handling import XY_2_X_Y
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

get_unlabeled_data(load=False)
_, test_Y = XY_2_X_Y(pd.read_csv('ElectionsData_Pred_Features for check test.csv'))
print(type(test_Y))
test_Y = test_Y.map({
        'Khakis': 0, 'Oranges': 1, 'Purples': 2, 'Turquoises': 3, 'Yellows': 4, 'Blues': 5, 'Whites': 6,
        'Greens': 7, 'Violets': 8, 'Browns': 9, 'Greys': 10, 'Pinks': 11, 'Reds': 12,
    })
train_X, train_Y = XY_2_X_Y(train_XY)

ab = AdaBoostClassifier(RandomForestClassifier(n_estimators=85))
ab.fit(train_X, train_Y)
print(test_Y)
print(accuracy_score(ab.predict(test_X), test_Y))
