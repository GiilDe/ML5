import numpy as np


def coalition_score(coalition_, XY):
    coalition = XY[XY['Vote'].isin(coalition_)].to_numpy()
    opposition = XY[~XY['Vote'].isin(coalition_)].to_numpy()
    coalition = coalition[:, 1:]
    opposition = opposition[:, 1:]
    avg_coalition = np.average(coalition, axis=0)
    avg_opposition = np.average(opposition, axis=0)
    avg_dist = np.average([np.linalg.norm(x - avg_coalition, ord=2) for x in coalition])
    normalized_avg_dist = (avg_dist - 0.9082616340543562)/(0.9416345219593429 - 0.9082616340543562)
    normalized_coal_opp_dist = (np.linalg.norm(avg_coalition - avg_opposition, ord=2)-0.11884543093915637)/(0.7848325253103648 - 0.11884543093915637)
    return (normalized_coal_opp_dist + 1)/(normalized_avg_dist + 1)


def unnormalized_coalition_score(coalition_, XY):
    coalition = XY[XY['Vote'].isin(coalition_)].to_numpy()
    opposition = XY[~XY['Vote'].isin(coalition_)].to_numpy()
    coalition = coalition[:, 1:]
    opposition = opposition[:, 1:]
    avg_coalition = np.average(coalition, axis=0)
    avg_opposition = np.average(opposition, axis=0)
    avg_dist = np.average([np.linalg.norm(x - avg_coalition, ord=2) for x in coalition])
    coal_opp_dist = np.linalg.norm(avg_coalition - avg_opposition, ord=2)
    return coal_opp_dist/avg_dist