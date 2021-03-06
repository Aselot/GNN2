import numpy as np

np.random.seed(0)


def create_data(points, classes):
    X = np.zeros((points * classes, 2))  # list of given number of points per each class, containing pairs of values
    y = np.zeros(points * classes, dtype='uint8')  # same as above, but containing simple values - classes
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))  # index in class
        X[ix] = np.c_[np.random.randn(points) * .1 + (class_number) / 3, np.random.randn(points) * .1 + 0.5]
        y[ix] = class_number
    return X, y
