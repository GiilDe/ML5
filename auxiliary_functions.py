import matplotlib.pyplot as plt
import numpy as np


def plot_2d_predictions_borders_for_trained_clf(clf, X, Y):
    colors = ['b', 'g', 'r']
    colors_dict = dict([(label, color) for label, color in zip(list(set(Y)), colors[0: len(list(set(Y)))])])
    X_ = []
    for x_1 in np.arange(min([x[0] for x in X]), max([x[0] for x in X]),
                         (max([x[0] for x in X]) - min([x[0] for x in X])) / 70):
        for x_2 in np.arange(min([x[1] for x in X]), max([x[1] for x in X]),
                             (max([x[1] for x in X]) - min([x[1] for x in X])) / 70):
            X_.append(np.array([x_1, x_2]))
    X_ = np.array(X_)
    print('here 1')
    plt.scatter([x[0] for x in X_], [x[1] for x in X_], c=[colors_dict[label] for label in clf.predict(X_)])
    print('here 2')
    colors = ['c', 'm', 'y', 'k', 'w']
    print('here 3')
    colors_dict = dict([(label, color) for label, color in zip(list(set(Y)), colors[0: len(list(set(Y)))])])
    plt.scatter([x[0] for x in X], [x[1] for x in X], c=[colors_dict[label] for label in Y])
    print('here 4')
    plt.show()


def plot_scatter(Xs, Ys, title=''):
    plt.title(title)
    plt.scatter(Xs, Ys)
    plt.show()


def plot_data(Xs, Ys, title=''):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    Ys_Xs_zip = list(zip(Ys, Xs))
    Ys_Xs_zip.sort(key=lambda x: x[0])
    Xs_by_y = dict()

    for label, point in Ys_Xs_zip:
        if Xs_by_y.get(label) is None:
            Xs_by_y[label] = [point]
        else:
            Xs_by_y[label].append(point)

    color_index = 0
    for label in Xs_by_y.keys():
        Xs = Xs_by_y[label]
        X = []
        Y = []
        for x in Xs:
            X.append(x[0])
            Y.append(x[1])
        plt.scatter(X, Y, c=colors[color_index])
        color_index += 1
    plt.show()


def get_y_x_num_gradient(X, Y, f, gradient, f_params,):
    y_x_num_gradient = []
    for i in range(len(X)):
        y_x_num_gradient.append((Y[i], X[i], f(X[i], f_params), gradient(X[i], f_params)))
    return y_x_num_gradient

def get_numbers_by_order(X, f, f_params, with_Xs=False):
    if with_Xs:
        return [(f(x, f_params), x) for x in X]
    else:
        return [f(x, f_params) for x in X]


min_for_new_bin = 0
def long_enough_row(numbers_Y_zip):
    # print('here')
    for i in range(min_for_new_bin):
        if len(numbers_Y_zip) <= i + 1:
            return False
        if numbers_Y_zip[i][1] != numbers_Y_zip[i + 1][1]:
            return False
    return True


def get_m(numbers, Y):
    numbers_Y_zip = list(zip(numbers, Y))
    numbers_Y_zip.sort(key=lambda x: x[0])
    m = 0
    current_y = numbers_Y_zip[0][1]
    for num, y in numbers_Y_zip[1:]:
        if y != current_y and long_enough_row(numbers_Y_zip[current_index:]):
            m += 1
    # m += 1
    return m


def get_bins(numbers_Y_zip):
    numbers_Y_zip.sort(key=lambda x: x[0])

    bins = []  # list of (last of prev label, first of new_label, new_label)
    prev_label = numbers_Y_zip[0][1]
    bin_to_label = []
    times_of_label_in_a_row = 0
    for num, label in numbers_Y_zip:
        if label != prev_label:
            bins.append((num + prev_num)/2)
            times_of_label_in_a_row = 1
            bin_to_label.append(prev_label)
        else:
            times_of_label_in_a_row += 1
        prev_num = num
        prev_label = label

    bins.append(float('inf'))
    bin_to_label.append(prev_label)
    return bins, bin_to_label


def plot_colored_scatter_by_bins(Xs, Ys, f, f_params, title=''):
    numbers_Y_zip = []
    numbers = []
    for x, y in zip(Xs, Ys):
        number = f(x, f_params)
        numbers.append(number)
        numbers_Y_zip.append((number, y))
    bins, bin_to_label = get_bins(numbers_Y_zip)

    X_ = []
    for x_1 in np.arange(min([x[0] for x in Xs]), max([x[0] for x in Xs]),
                         (max([x[0] for x in Xs]) - min([x[0] for x in Xs])) / 100):
        for x_2 in np.arange(min([x[1] for x in Xs]), max([x[1] for x in Xs]),
                             (max([x[1] for x in Xs]) - min([x[1] for x in Xs])) / 100):
          X_.append(np.array([x_1, x_2]))

    colors = ['b', 'g', 'r', 'w']
    colors_dict = dict([(label, color) for label, color in zip(list(set(Ys)), colors[0: len(list(set(Ys)))])])

    # print(bins)
    numbers_ = []
    for x in X_:
        numbers_.append(f(x, f_params))

    # print('numbers_: ', numbers_)
    # print('bins: ', bins)

    prediction = [bin_to_label[i] for i in np.digitize(numbers_, bins)]
    plt.scatter([x[0] for x in X_], [x[1] for x in X_], c=[colors_dict[label] for label in prediction])

    colors = ['c', 'm', 'y', 'k']
    colors_dict = dict([(label, color) for label, color in zip(list(set(Ys)), colors[0: len(list(set(Ys)))])])
    plt.scatter([x[0] for x in Xs], [x[1] for x in Xs], c=[colors_dict[label] for label in Ys])
    plt.show()
