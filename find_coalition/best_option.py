from itertools import chain, combinations
import pandas as pd
from find_coalition.coalition_score import coalition_score
from data_handling import X_Y_2_XY
from get_prepared_data import get_prepared_data


def big_enough_coalition(parties_list, test_X, test_Y_hat):
    coalition = test_X[test_Y_hat.isin(parties_list)]
    return len(coalition) >= len(test_X) * 0.51


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def best_option_ever(X, Y):
    best_coalition_score = float('-inf')
    best_coalition = None
    # counter = 0
    for coalition in powerset(set(range(0, 12))):
        # print(counter)
        # counter += 1
        if not big_enough_coalition(coalition, X, Y):
            continue
        else:
            score = coalition_score(coalition, X_Y_2_XY(X, Y))
            if score > best_coalition_score:
                best_coalition_score = score
                best_coalition = coalition
    return best_coalition, best_coalition_score


def get_coalition(X, Y):
    print(best_option_ever(X, Y))

