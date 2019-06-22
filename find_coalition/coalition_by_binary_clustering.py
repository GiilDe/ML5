from get_prepared_data import get_prepared_data
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from coalition_by_clustering import get_party_cluster_hist
from data_handling import X_Y_2_XY
import itertools
from coalition_score import coalition_score


train_X, train_Y, validation_X, validation_Y, test_X, test_Y = get_prepared_data(load=True)

X_to_split = pd.concat([train_X, validation_X])
Y_to_split = pd.concat([train_Y, validation_Y])


def scorer(estimator, X, Y):
    clusters = estimator.predict(X)

    cluster_list = [[], []]

    for party in set(Y):
        party_culsters = clusters[Y == party]
        _, counts = np.unique(party_culsters, return_counts=True)
        cluster_list[np.argmax(counts)].append(party)

    cluster_0_size = len(X[Y.isin(cluster_list[0])])
    cluster_1_size = len(X[Y.isin(cluster_list[1])])

    bigger_cluster = np.argmax([cluster_0_size, cluster_1_size])
    coalition = cluster_list[bigger_cluster]
    score = coalition_score(coalition, X_Y_2_XY(X, Y))

    return score


best_initialization = None
best_score = float('-inf')
for _ in range(20):
    initialization = np.random.normal(0.5, 1, (2, len(train_X.columns)))
    kmean = KMeans(2, initialization)
    score = np.average(cross_val_score(kmean, X_to_split, Y_to_split, cv=3, scoring=scorer))
    if score > best_score:
        best_score = score
        best_initialization = initialization


kmean = KMeans(2, best_initialization)
kmean.fit(X_to_split, Y_to_split)
clusters = kmean.predict(test_X)

cluster_list = [[], []]
for party in set(test_Y):
    party_culsters = clusters[test_Y == party]
    _, counts = np.unique(party_culsters, return_counts=True)
    cluster_list[np.argmax(counts)].append(party)

cluster_0_size = len(test_X[test_Y.isin(cluster_list[0])])
cluster_1_size = len(test_X[test_Y.isin(cluster_list[1])])

bigger_cluster = np.argmax([cluster_0_size, cluster_1_size])

coalition = cluster_list[bigger_cluster]

print(coalition)
print(coalition_score(coalition, X_Y_2_XY(test_X, test_Y)))
