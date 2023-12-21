from utils_plot import *
import math


def sigmoid_function(z):
    g = 1/(1 + np.exp(-z))
    return g


def compute_cost_without_reg(x_train, y_train, w, b):
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


def compute_cost_with_reg(x_train, y_train, w_arr, b_in, lambda_ = 1):
    m, n = x_train.shape

    cost_without_reg = compute_cost_without_reg(x_train, y_train, w_arr, b_in)

    reg_cost = 0.

    for j in range(n):
        reg_cost_j = w_arr[j]**2
        reg_cost += reg_cost_j

    cost_with_reg = cost_without_reg + (lambda_/(2*m)) * reg_cost

    return cost_with_reg


def compute_gradient_without_reg(x_train, y_train, w_arr, b):
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


def compute_gradient_reg(x_train, y_train, w_arr, b_in, lambda_):
    m, n = x_train.shape

    dj_dw, dj_db = compute_gradient_without_reg(x_train, y_train, w_arr, b_in)

    for j in range(n):
        dj_dw_j_reg = (lambda_/m) * w_arr[j]
        dj_dw[j] += dj_dw_j_reg

    return dj_dw, dj_db


def gradient_descent(x_train, y_train, w_arr_in, b_in, alpha, num_iters, lambda_):

    j_history = []
    w_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_reg(x_train, y_train, w_arr_in, b_in, lambda_)

        w_arr_in = w_arr_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db

        if i < 100000:
            cost = compute_cost_with_reg(x_train, y_train, w_arr_in, b_in)
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

        p[i] = fw_b >= 0.5

    return p


x_train, y_train = load_dataset_from_file("datasetRLR")
plot_data_RLR(x_train, y_train)
mapped_x_train = map_feature(x_train[:, 0], x_train[:, 1])

np.random.seed(1)
initial_w = np.random.rand(mapped_x_train.shape[1]) - 0.5
initial_b = 1.

lambda_ = 0.01
iterations = 100000
alpha = 0.01

w, b, J_history, W_history = gradient_descent(mapped_x_train, y_train, initial_w, initial_b, alpha, iterations, lambda_)
x_train2 = load_dataset_from_file('datasetRLR2')
x_train2 = map_feature(x_train2[:, 0], x_train2[:, 1])
predict = predict(x_train2, w, b)
print(f'Train Accuracy by my test: {np.mean(predict == y_train) * 100}%')