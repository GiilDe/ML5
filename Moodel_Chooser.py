from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter
import ast
from get_prepared_data import get_prepared_data
from MatrixForest import MatrixForest

class Model_Chooser:
    def __init__(self, classifiers_dict=None, score_measurement_for_find_best_params='accuracy', predict_proba=False):
        if classifiers_dict is None:
            if not predict_proba:
                self.classifiers_dict = {  # key is classifier name, value is a list of tuples, where first val is parameter
                                           # name for the model and second val is a list of possible values for the parameter
                    # 'KNeighborsClassifier': [('n_neighbors', list(range(1, 15, 2)))],  # predict_proba
                    'SVC': [('kernel', ['linear', 'poly', 'rbf', 'sigmoid']), ('gamma', ['scale'])],
                     # 'DecisionTreeClassifier': [('min_samples_split ', list(range(2, 52, 5)))],  # predict_proba
                    'RandomForestClassifier': [('n_estimators', list(range(5, 100, 20))),  # predict_proba
                                               ('min_samples_split', list(range(2, 52, 5)))],
                    'GaussianNB': [],  # predict_proba
                    # 'LinearDiscriminantAnalysis': [],
                    # 'MatrixForest': [('num_matrix', np.arange(0, 50, 5)), ('h', np.linspace(0.0001, 1, 20))],
                    # 'LogisticRegression': [('solver', ['lbfgs'])],  # predict_proba
                    #  'QuadraticDiscriminantAnalysis': [],  # predict_proba
                    }
            else:
                self.classifiers_dict = {  # key is classifier name, value is a list of tuples, where first val is parameter
                    # name for the model and second val is a list of possible values for the parameter
                    'KNeighborsClassifier': [('n_neighbors', list(range(1, 15, 2)))],  # predict_proba
                    'DecisionTreeClassifier': [('min_samples_split ', list(range(2, 52, 5)))],  # predict_proba
                    'RandomForestClassifier': [('n_estimators', list(range(5, 20, 5))),  # predict_proba
                                               ('min_samples_split', list(range(2, 52, 5)))],
                    'GaussianNB': [],  # predict_proba,
                    'LinearDiscriminantAnalysis': [],
                    'LogisticRegression': [('solver', ['lbfgs'])],  # predict_proba
                    'QuadraticDiscriminantAnalysis': [],  # predict_proba
                }
            self.classifiers_params_dict = dict()

    def create_classifier(self, classifier, param_list):
        if len(param_list) == 0:
            clf = eval(classifier + '()')
        elif len(param_list) == 1:
            param_name = param_list[0][0]
            param_val = param_list[0][1]
            clf = eval(classifier + '(' + param_name + '=' + str(param_val) + ')')
        elif len(param_list) == 2:
            first_param_name = param_list[0][0]
            first_param_val = param_list[0][1]
            second_param_name = param_list[1][0]
            second_param_val = param_list[1][1]
            clf = eval(classifier + '(' + first_param_name + '=' + str(first_param_val) + ', ' +
                       second_param_name + '=' + str(second_param_val) + ')')
        else:
            print('Param number greater than 2 is unsupported at the moment'.capitalize())
            exit()
        return clf

    def find_classifiers_best_params(self, X, Y, score_measure='accuracy'):
        for classifier, param_list in self.classifiers_dict.items():
            if len(param_list) == 0:
                self.classifiers_params_dict[classifier] = []
            elif len(param_list) == 1:
                best_score = 0
                best_param = None
                param_name = param_list[0][0]
                for param_val in param_list[0][1]:
                    if isinstance(param_val, str):
                        param_val = '\'' + param_val + '\''
                    clf = eval(classifier + '(' + param_name + '=' + str(param_val) + ')')
                    score = np.average(cross_validate(clf, X, Y, scoring=score_measure, cv=3)['test_score'])
                    if score > best_score:
                        best_score = score
                        best_param = param_val
                self.classifiers_params_dict[classifier] = [(param_name, best_param)]
            elif len(param_list) == 2:
                best_score = 0
                best_first_param = None
                best_second_param = None
                first_param_name = param_list[0][0]
                second_param_name = param_list[1][0]
                for first_param_val in param_list[0][1]:
                    if isinstance(first_param_val, str):
                        first_param_val = '\'' + first_param_val + '\''
                    for second_param_val in param_list[1][1]:
                        if isinstance(second_param_val, str):
                            second_param_val = '\'' + second_param_val + '\''
                        clf = eval(classifier + '(' + first_param_name + '=' + str(first_param_val) + ', ' +
                                   second_param_name + '=' + str(second_param_val) + ')')
                        score = np.average(cross_validate(clf, X, Y, cv=3, scoring='accuracy')['test_score'])
                        if score > best_score:
                            best_score = score
                            best_first_param = first_param_val
                            best_second_param = second_param_val
                self.classifiers_params_dict[classifier] = [(first_param_name, best_first_param),
                                                            (second_param_name, best_second_param)]
            else:
                print('Param number greater than 2 is unsupported at the moment'.capitalize())
        return self.classifiers_params_dict

    def get_winner(self, train_X, train_Y, test_X, test_Y, score_measure_func=None):
        scores_list = []
        for classifier, param_list in self.classifiers_params_dict.items():
            clf = self.create_classifier(classifier, param_list)
            clf.fit(train_X, train_Y)
            prediction = clf.predict(test_X)
            if score_measure_func is not None:
                score = score_measure_func(prediction, test_Y)
            else:
                score = accuracy_score(prediction, test_Y)
            scores_list.append((score, classifier, param_list))
        scores_list.sort(key=lambda x: x[0], reverse=True)
        return scores_list