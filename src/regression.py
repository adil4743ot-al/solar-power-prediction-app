import numpy as np
def hypothesis(X, theta):
    return X @ theta
def cost(X, y, theta):
    prediction = hypothesis(X, theta)
    error = prediction - y
    return 0.5 * np.sum(error ** 2)
def fit_normal(X, y):
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta
def fit_batch_gd(X, y, alpha, n_iters):
    theta = np.zeros(X.shape[1])
    cost_history = []
    for i in range(n_iters):
        prediction = hypothesis(X, theta)
        error = y - prediction
        theta = theta + alpha * (X.T @ error)
        cost_history.append(
            cost(X, y, theta)
        )
    return theta, cost_history
def fit_sgd(X, y, alpha, n_epochs):
    theta = np.zeros(X.shape[1])
    cost_history = []
    for epoch in range(n_epochs):
        for i in range(len(y)):
            prediction = X[i] @ theta
            error = y[i] - prediction
            theta = theta + alpha * error * X[i]
        cost_history.append(
            cost(X, y, theta)
        )
    return theta, cost_history
def rmse(y, prediction):
    error = y - prediction
    return np.sqrt(
        np.mean(error ** 2)
    )
