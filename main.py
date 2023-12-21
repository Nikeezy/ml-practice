from utils_plot import *
import math


def predict_check(x_train, p_arr):
    predict_arr = np.array

    for i in range(len(x_train)):
        if sum(x_train[i]) >= 110:
            predict_arr = np.append(predict_arr, 1)
        else:
            predict_arr = np.append(predict_arr, 0)

    predict_arr = predict_arr[1:predict_arr.shape[0]]
    print(f'Train Accuracy by my test: {np.mean(predict_arr == p_arr) * 100}%')


def sigmoid_function(z):
    g = 1/(1 + np.exp(-z))
    return g


def compute_cost(x_train, y_train, w, b):
    m, n = x_train.shape
    loss_sum = 0.

    for i in range(m):
        z_wb = 0.
        for j in range(n):
            z_wb_ij = w[j] * x_train[i][j]
            z_wb += z_wb_ij
        z_wb += b

        f_wb = sigmoid_function(z_wb)

        loss = -y_train[i]*np.log(f_wb)-(1-y_train[i])*np.log(1 - f_wb)

        loss_sum += loss

    total_cost = loss_sum/m
    return total_cost


def compute_gradient(x_train, y_train, w_arr, b):
    m, n = x_train.shape
    dj_dw = np.zeros(w_arr.shape)
    dj_db = 0.

    for i in range(m):

        f_wb = np.dot(x_train[i], w_arr) + b

        dj_db_i = f_wb - y_train[i]

        dj_db += dj_db_i

        for j in range(n):
            dj_dw_ij = (f_wb - y_train[i])*x_train[i][j]
            dj_dw[j] += dj_dw_ij

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db


def gradient_descent(x_train, y_train, w_arr_in, b_in, alpha, num_iters, lambda_):
    j_history = []
    w_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x_train, y_train, w_arr_in, b_in)

        w_arr_in = w_arr_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db

        if i < 100000:
            cost = compute_cost(x_train, y_train, w_arr_in, b_in)
            j_history.append(cost)

        if i % math.ceil(num_iters / 10) == 0 or i == (num_iters - 1):
            w_history.append(w_arr_in)
            print(w_history[-1])
            print(f"Iteration {i:4}: Cost {float(j_history[-1]):8.2f}   ")

    return w_arr_in, b_in, j_history, w_history


def predict(x_train, w, b):
    m, n = x_train.shape
    p = np.zeros(m)

    for i in range(m):
        z_wb = 0
        for j in range(n):
            z_wb_ij = w[j] * x_train[i][j]
            z_wb += z_wb_ij

        z_wb += b

        fw_b = sigmoid_function(z_wb)

        p[i] = fw_b >= 0.6

    return p


generate_train_data_set_LR(1000, 10000)

x_train, y_train = load_dataset_from_file("dataset")

np.random.seed(1)
initial_w = 0.001 * (np.random.rand(2).reshape(-1, 1) - 0.5)
initial_b = -8

iterations = 100000
alpha = 0.000001

w, b, j_history, w_history = gradient_descent(x_train, y_train, initial_w, initial_b, alpha, iterations, 0)
x_train2 = load_dataset_from_file("dataset2")
predict_result = predict(x_train2, w, b)

for i in range(len(x_train2)):
    print(f"{i + 1} train_example: {x_train2[i]} - {predict_result[i]}", end='\n')

predict_check(x_train2, predict_result)
