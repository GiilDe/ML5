
from get_prepared_data import get_prepared_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from get_prepared_data import get_prepared_data_test

train_X, train_Y, validation_X, validation_Y, test_X, test_Y = get_prepared_data_test()


cl = RandomForestClassifier(n_estimators=85)
ad = AdaBoostClassifier(cl)

