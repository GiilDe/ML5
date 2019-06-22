from Moodel_Chooser import Model_Chooser
from collections import Counter
from get_prepared_data import get_prepared_data
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from MatrixForest import Check, MatrixForest
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier


train_X, train_Y, validation_X, validation_Y, test_X, test_Y = get_prepared_data()
X_to_split = pd.concat([train_X, validation_X])
Y_to_split = pd.concat([train_Y, validation_Y])

# clf = MatrixForest(num_matrix=1000, h=0.9)
# clf.fit(X_to_split.to_numpy(), Y_to_split.to_numpy())
# print(accuracy_score(clf.predict(test_X.to_numpy()), test_Y.to_numpy()))


# mlp = MLPClassifier(hidden_layer_sizes=[200, 74, 146, 180, 54], activation='tanh', verbose=True, max_iter=500, early_stopping=False)
# mlp.fit(train_X, train_Y)
# Y_hat = mlp.predict(validation_X)
# accuracy = accuracy_score(validation_Y, Y_hat)


mc = Model_Chooser()
mc.find_classifiers_best_params(X_to_split, Y_to_split)

scores_list = mc.get_winner(X_to_split, Y_to_split, test_X, test_Y)
print(scores_list)

RF = RandomForestClassifier(n_estimators=85, random_state=0)
RF.fit(X_to_split, Y_to_split)
print(accuracy_score(RF.predict(test_X), test_Y))


classifiers = [('clf ' + str(i), AdaBoostClassifier(mc.create_classifier(scores_list[i][1], scores_list[i][2]))) for i in range(3)]
classifiers.append(('nn', MLPClassifier(hidden_layer_sizes=[200, 74, 146, 180, 54], activation='tanh', verbose=True,
                                        max_iter=500, early_stopping=False)))
weights = [x[0]**2 for x in scores_list[:3]]
weights.append(scores_list[0][0]**2)
eclf1 = VotingClassifier(estimators=classifiers, weights=weights)
#
eclf1.fit(X_to_split, Y_to_split)
print(accuracy_score(eclf1.predict(test_X.to_numpy()), test_Y.to_numpy()))







# bclf = Check(h=1)
# bclf.fit(X_to_split, Y_to_split)
# print(accuracy_score(bclf.predict(test_X), test_Y))

# best_classifier = mc.create_classifier(best_classifier, best_param_list)
# print('best_classifier: ', best_classifier)
# print('best_param_list: ', best_param_list)
# best_classifier.fit(X_to_split, Y_to_split)
# prediction = list(best_classifier.predict(test_X))
# print('Accuracy: '.capitalize(), accuracy_score(prediction, test_Y))

