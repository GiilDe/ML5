from data_handling import to_numerical_data, split_data, scale_all, X_Y_2_XY, XY_2_X_Y
from imputations import impute_train_X, impute_test_and_validation
import pandas as pd
from os.path import isfile


def get_data():
    data = pd.read_csv('ElectionsData.csv')
    data = data.loc[:, ['Vote', 'Avg_environmental_importance', 'Avg_government_satisfaction',
                        'Avg_education_importance', 'Most_Important_Issue', 'Avg_monthly_expense_on_pets_or_plants',
                        'Avg_Residancy_Altitude', 'Yearly_ExpensesK', 'Weighted_education_rank',
                        'Number_of_valued_Kneset_members']]
    return data


def get_prepared_data():
    if isfile('train_XY.csv'):
        train_XY = pd.read_csv('train_XY.csv')
        validation_XY = pd.read_csv('validation_XY.csv')
        test_XY = pd.read_csv('test_XY.csv')
        print('\033[1m' + "DATA LOADED" + '\033[0m')
    else:
        print('\033[1m' + "PREPARING DATA" + '\033[0m')
        data = get_data()
        data = data.iloc
        data = to_numerical_data(data)
        train_X, train_Y, validation_X, validation_Y, test_X, test_Y = split_data(data)
        train_XY = X_Y_2_XY(train_X, train_Y)
        validation_XY = X_Y_2_XY(validation_X, validation_Y)
        test_XY = X_Y_2_XY(test_X, test_Y)
        train_XY = impute_train_X(train_XY)
        # train_XY = clean_and_correct_train_X(train_XY)
        train_XY, validation_XY, test_XY = scale_all(train_XY, validation_XY, test_XY)
        validation_XY, test_XY = impute_test_and_validation(train_XY, validation_XY, test_XY)
        train_XY.to_csv('train_XY.csv')
        validation_XY.to_csv('validation_XY.csv')
        test_XY.to_csv('test_XY.csv')
        print('\033[1m' + "DATA SAVED" + '\033[0m')
    train_X, train_Y = XY_2_X_Y(train_XY)
    validation_X, validation_Y = XY_2_X_Y(validation_XY)
    test_X, test_Y = XY_2_X_Y(test_XY)
    return train_X, train_Y, validation_X, validation_Y, test_X, test_Y

def get_unlabeled_data():
    data = pd.read_csv('ElectionsData.csv')