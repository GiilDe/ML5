from sklearn.neural_network import MLPClassifier
from get_prepared_data import get_prepared_data
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from get_prepared_data import get_prepared_data_test
from sklearn.metrics import accuracy_score
import pickle as pk
import itertools


train_X, train_Y, validation_X, validation_Y, test_X, test_Y = get_prepared_data()


# models = []
# for n_classifiers, criterion in itertools.product(range(40, 100, 2), ):
#     cl = RandomForestClassifier(n_estimators=85, criterion=criterion)
#     ad = AdaBoostClassifier(cl, n_estimators=n_classifiers)
#     ad.fit(train_X, train_Y)
#     validation_Y_hat = ad.predict(validation_X)
#     accuracy = accuracy_score(validation_Y, validation_Y_hat)
#     models.append(((n_classifiers, criterion), ad))
#     print(str(n_classifiers) + " " + criterion + " " + str(accuracy))

# ad = AdaBoostClassifier(cl, n_estimators=29)
# ad.fit(train_X, train_Y)
# validation_Y_hat = ad.predict(validation_X)
# accuracy = accuracy_score(validation_Y, validation_Y_hat)
#print(str(n_classifiers) + " " + str(accuracy))


mlp = MLPClassifier(hidden_layer_sizes=[200, 74, 146, 180, 54], activation='tanh', verbose=True, max_iter=500, early_stopping=True)
random_forest = pk.load(open("model", 'rb'))
ens_classifier = VotingClassifier([("net", mlp), ("forest", random_forest)])
ens_classifier.fit(train_X, train_Y)
Y_hat = ens_classifier.predict(validation_X)
accuracy = accuracy_score(validation_Y, Y_hat)
