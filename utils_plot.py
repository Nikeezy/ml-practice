import random
import numpy as np
import matplotlib.pyplot as plt


def load_dataset_from_file(filename):
    file = open(filename, "r")
    data_string = file.readline()

    x_train = np.ndarray
    y_train = np.ndarray

    count = 0

    check = False

    while data_string != '':
        data_string = data_string.split(' ')
        if len(data_string) == 3:

            data_string = list(map(float, data_string))

            x_data = data_string[:len(data_string) - 1]
            y_data = data_string[len(data_string) - 1]

            x_train = np.append(x_train, x_data)
            y_train = np.append(y_train, y_data)

        elif len(data_string) == 2:

            check = True

            data_string = list(map(float, data_string))

            x_data = data_string

            x_train = np.append(x_train, x_data)

        count += 1
        data_string = file.readline()

    if check:
        x_train = x_train[1:x_train.shape[0]].astype(float)

        x_train = x_train.reshape(count, 2)
        return x_train
    else:
        x_train = x_train[1:x_train.shape[0]].astype(float)
        y_train = y_train[1:y_train.shape[0]]

        x_train = x_train.reshape(count, 2)
        return x_train, y_train


def plot_data_LR(x_train, y_train, pos_label="Admitted", neg_label="Not admitted"):
    plt.ylabel('Exam 2 score')
    plt.xlabel('Exam 1 score')

    plt.plot(100, 100, 'X', c="red", label=neg_label)
    plt.plot(100, 100, 'o', c="green", label=pos_label)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')

    for i in range(len(y_train)):
        if y_train[i] == 1:
            plt.scatter(x_train[i, 0], x_train[i, 1], c="green", s=100)
        elif y_train[i] == 0:
            plt.scatter(x_train[i, 0], x_train[i, 1], marker="X", c="red", s=100)

    plt.show()


def plot_predict_LR(x_train, p_arr, pos_label="Admitted", neg_label="Not admitted"):
    plt.ylabel('Exam 2 score')
    plt.xlabel('Exam 1 score')

    plt.plot(100, 100, 'X', c="red", label=neg_label)
    plt.plot(100, 100, 'o', c="green", label=pos_label)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')

    for i in range(len(p_arr)):
        if p_arr[i] == 1:
            plt.scatter(x_train[i, 0], x_train[i, 1], c="green", s=100)
        elif p_arr[i] == 0:
            plt.scatter(x_train[i, 0], x_train[i, 1], marker="X", c="red", s=100)

    plt.show()


def generate_train_data_set_LR(n, m):
    file = open("dataset", "w")

    i = 0

    while i < n / 2:
        num1 = random.uniform(0, 100)
        num2 = random.uniform(0, 100)
        if num1 + num2 >= 120:
            file.write(format(num1, '.8f') + ' ' + format(num2, '.8f') + ' 1\n')
            i += 1

    j = 0

    while j < n / 2:
        num1 = random.uniform(0, 100)
        num2 = random.uniform(0, 100)
        if num1 + num2 < 120:
            file.write(format(num1, '.8f') + ' ' + format(num2, '.8f') + ' 0\n')
            j += 1

    file.close()

    file = open("dataset2", "w")

    for i in range(m):
        num1 = random.uniform(0, 100)
        num2 = random.uniform(0, 100)
        file.write(format(num1, '.8f') + ' ' + format(num2, '.8f') + '\n')

    file.close()


def map_feature(X1, X2):
    degree = 6
    out = np.ones(( X1.shape[0], sum(range(degree + 2))))
    curr_column = 1
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out[:, curr_column] = np.power(X1, i-j) * np.power(X2, j)
            curr_column += 1

    return out


def plot_data_RLR(x_train, y_train, pos_label="Accepted", neg_label="Rejected"):
    plt.ylabel('Microchip Test 2')
    plt.xlabel('Microchip Test 1')

    plt.plot(1, 1, 'X', c="red", label=neg_label)
    plt.plot(1, 1, '+', c="green", label=pos_label)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')

    for i in range(len(y_train)):
        if y_train[i] == 1:
            plt.scatter(x_train[i, 0], x_train[i, 1], marker="+", c="green", s=100)
        elif y_train[i] == 0:
            plt.scatter(x_train[i, 0], x_train[i, 1], marker="X", c="red", s=100)

    plt.show()