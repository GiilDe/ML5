from party_num_to_name import inv_map
from get_prepared_data import get_unlabeled_data
from sklearn.metrics import davies_bouldin_score
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import pandas as pd
from matplotlib.pyplot import hist
import numpy as np


def find_best(range, scoring):
    best_score = None
    best_val = None
    for val in range:
        score = scoring(val)
        if best_score is None or score < best_score:
            best_val = val
            best_score = score

    return best_val


train, test = get_unlabeled_data(load=True)


def score_with_davies(k):
    kmean = KMeans(k)
    labels = kmean.fit_predict(test)
    score = davies_bouldin_score(test, labels)
    return score


best_k = find_best(range(2, 13), score_with_davies)
kmean_for_test = KMeans(best_k)
clusters = kmean_for_test.fit_predict(test)
parties = pd.read_csv("predictions.csv")
parties = parties.drop('IdentityCard_Num', axis=1)
parties['clusters_labels'] = pd.Series(clusters)
parties['predictions'] = parties['predictions'].map(inv_map)

clusters_hists = []
for i in range(0, 13):
    party_clusters_hist = np.zeros(8)
    for sample in parties.iterrows():
        n = sample[1]['predictions']
        if sample[1]['predictions'] == i:
            party_clusters_hist[sample[1]['clusters_labels']] += 1

    clusters_hists.append(party_clusters_hist)

