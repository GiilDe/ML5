from get_prepared_data import get_prepared_data
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import scipy as sp
from coalition_score import coalition_score as getCoalitionScore
from data_handling import X_Y_2_XY


def calc_sim(party1_cluster_hist, party2_cluster_hist):
    similarity, _ = sp.stats.pearsonr(party1_cluster_hist, party2_cluster_hist)
    return similarity + 1


def normalize(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
       return v
    return v / norm


def get_similarities(parties_num, get_similarity, cluster_labels, data_X: pd.DataFrame):
    parties_similarities = {}
    for i in range(parties_num):
        for j in range(i+1, parties_num):
            parties_similarities[frozenset([i, j])] = get_similarity(i, j, cluster_labels, data_X)
    return parties_similarities


def get_party_in_cluster_num(party, data, label_num, cluster_labels):
    if 'cluster_label' not in data.columns:
        data['cluster_label'] = cluster_labels
    return len(data[(data['Vote'] == party) & (data['cluster_label'] == label_num)])


def get_party_cluster_hist(party, cluster_labels, df: pd.DataFrame):
    N = len(np.unique(cluster_labels))
    party_cluster_hist = np.array([get_party_in_cluster_num(party, df, cluster, cluster_labels) for cluster in range(N)])
    return party_cluster_hist


def get_similarity(party1, party2, cluster_labels, df: pd.DataFrame):
    N = len(np.unique(cluster_labels))
    df = df.reset_index()
    cluster_labels = pd.Series(cluster_labels)
    df['cluster_label'] = cluster_labels

    party1_cluster_hist = get_party_cluster_hist(party1, cluster_labels, df)
    party2_cluster_hist = get_party_cluster_hist(party2, cluster_labels, df)

    similarity = calc_sim(party1_cluster_hist, party2_cluster_hist)

    return similarity


def most_similar_party(coalition):
    similarity_list = np.zeros(13)
    for party in coalition:
        for other_party in set(range(13)) - set(coalition):
            similarity_list[other_party] += similarity_matrix[frozenset([party, other_party])]

    assert len(coalition) == len(set(coalition))
    return np.argmax(similarity_list)


def big_enough_coalition(parties_list):
    coalition = test_X[test_Y.isin(parties_list)]
    return len(coalition) >= len(test_X) * 0.51


def build_coalition_from_similarity_matrix():
    best_coalition_score = float('-inf')
    best_coalition = list(range(13))
    for party in range(13):
        coalition = [party]
        while not big_enough_coalition(coalition):
            coalition.append(most_similar_party(coalition))
        coalition_score_ = getCoalitionScore(coalition, X_Y_2_XY(X_to_split, Y_to_split))
        if coalition_score_ > best_coalition_score:
            best_coalition_score = coalition_score_
            best_coalition = coalition.copy()
        while len(coalition) < 12:
            coalition.append(most_similar_party(coalition))
            coalition_score_ = getCoalitionScore(coalition, X_Y_2_XY(X_to_split, Y_to_split))
            if coalition_score_ > best_coalition_score:
                best_coalition_score = coalition_score_
                best_coalition = coalition.copy()

    best_coalition.sort()
    return best_coalition, best_coalition_score


train_X, train_Y, validation_X, validation_Y, test_X, test_Y = get_prepared_data()
X_to_split = pd.concat([train_X, validation_X])
Y_to_split = pd.concat([train_Y, validation_Y])
data_test = X_Y_2_XY(test_X, test_Y, False)
best_score = float('-inf')
best_coalition = None
best_k = None
for k in range(2, 50, 2):
    k_means = KMeans(n_clusters=k)
    cluster_res = k_means.fit(train_X)
    cluster_labels = k_means.predict(test_X)
    parties_num = len(np.unique(data_test['Vote']))
    similarities = get_similarities(parties_num, get_similarity, cluster_labels, data_test)
    similarity_matrix = similarities
    coalition, score = build_coalition_from_similarity_matrix()
    if score > best_score:
        best_score = score
        best_coalition = coalition
        best_k = k

