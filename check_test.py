from get_prepared_data import get_unlabeled_data
import pandas as pd
from data_handling import XY_2_X_Y
import pickle

abrf = pickle.load(open('model', 'r'))

tran_XY, test_X = get_unlabeled_data()
_, test_Y = XY_2_X_Y(pd.read_csv('for test check'))

