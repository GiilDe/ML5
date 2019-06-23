from sklearn.neural_network import MLPClassifier
from get_prepared_data import get_prepared_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from get_prepared_data import get_prepared_data_test
from sklearn.metrics import accuracy_score
import pickle as pk


train_X, train_Y, validation_X, validation_Y, test_X, test_Y = get_prepared_data()


cl = RandomForestClassifier(n_estimators=85)
models = []
for n_classifiers in range(15, 100, 2):
    ad = AdaBoostClassifier(cl, n_estimators=n_classifiers)
    ad.fit(train_X, train_Y)
    validation_Y_hat = ad.predict(validation_X)
    accuracy = accuracy_score(validation_Y, validation_Y_hat)
    models.append((n_classifiers, ad))
    print(str(n_classifiers) + " " + str(accuracy))

# ad = AdaBoostClassifier(cl, n_estimators=29)
# ad.fit(train_X, train_Y)
# validation_Y_hat = ad.predict(validation_X)
# accuracy = accuracy_score(validation_Y, validation_Y_hat)
#print(str(n_classifiers) + " " + str(accuracy))

mlp = MLPClassifier(hidden_layer_sizes=[200, 74, 146, 180, 54], activation='tanh', verbose=True, max_iter=500, early_stopping=False)
