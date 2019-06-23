import find_coalition.best_option
from get_prepared_data import get_unlabeled_data
import pickle as pk
import pandas as pd

# use brute force to find coalition on the test
train, test = get_unlabeled_data(load=True)
model = pk.load(open("model", 'rb'))
Y_hat = pd.Series(model.predict(test))
find_coalition.best_option.get_coalition(test, Y_hat)

