from get_prepared_data import get_prepared_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


train_X, train_Y, validation_X, validation_Y, test_X, test_Y = get_prepared_data()


mlp = MLPClassifier(hidden_layer_sizes=(4, 3,), verbose=True)
mlp.fit(train_X, train_Y)
Y_hat = mlp.predict(validation_X)
accuracy = accuracy_score(validation_Y, Y_hat)