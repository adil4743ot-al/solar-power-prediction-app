from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from regression import (
    hypothesis,
    fit_normal,
    fit_batch_gd,
    fit_sgd,
    rmse
)
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RESULTS = BASE / "results"
FIGURES = RESULTS / "figures"
SAVED_WEIGHTS = RESULTS / "saved_weights"
TABLES = RESULTS / "tables"
FIGURES.mkdir(parents=True, exist_ok=True)
SAVED_WEIGHTS.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)
plant = pd.read_csv(
    DATA / "plant1 hourly.csv"
)
weather = pd.read_csv(
    DATA / "plant1 openmeteo.csv"
)
plant["datetime"] = pd.to_datetime(
    plant["datetime"]
)
weather["datetime"] = pd.to_datetime(
    weather["datetime"]
)
data = pd.merge(
    plant,
    weather,
    on="datetime",
    how="inner"
)
print("Data shape:")
print(data.shape)
data["hour"] = data["datetime"].dt.hour
data["sin_hour"] = np.sin(
    2 * np.pi * data["hour"] / 24
)
data["cos_hour"] = np.cos(
    2 * np.pi * data["hour"] / 24
)
train = data[
    data["datetime"] < "2020-06-11"
].copy()

test = data[
    data["datetime"] >= "2020-06-11"
].copy()
print("Training rows:", len(train))
print("Testing rows:", len(test))
features_A = [
    "IRRADIATION",
    "MODULE_TEMPERATURE",
    "AMBIENT_TEMPERATURE",
    "sin_hour",
    "cos_hour"
]
features_B = [
    "sw_radiation",
    "temp_2m",
    "cloud_cover",
    "sin_hour",
    "cos_hour"
]
def make_X(train_data, test_data, features):

    mean = train_data[
        features
    ].mean().to_numpy()
    std = train_data[
        features
    ].std(
        ddof=0
    ).to_numpy()
    std[std == 0] = 1
    train_scaled = (
        train_data[
            features
        ].to_numpy() - mean
    ) / std
    test_scaled = (
        test_data[
            features
        ].to_numpy() - mean
    ) / std
    X_train = np.column_stack(
        (
            np.ones(len(train_scaled)),
            train_scaled
        )
    )
    X_test = np.column_stack(
        (
            np.ones(len(test_scaled)),
            test_scaled
        )
    )
    return X_train, X_test
XA, XAt = make_X(
    train,
    test,
    features_A
)
XB, XBt = make_X(
    train,
    test,
    features_B
)
y_train = train[
    "AC_POWER"
].to_numpy()
y_test = test[
    "AC_POWER"
].to_numpy()
batch_alphas = [
    1e-5,
    1e-4,
    1e-3
]
plt.figure(figsize=(8, 5))
for alpha in batch_alphas:

    theta, history = fit_batch_gd(
        XA,
        y_train,
        alpha,
        500
    )
    plt.plot(
        history,
        label="alpha = " + str(alpha)
    )
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title(
    "Batch Gradient Descent Learning Rates"
)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    FIGURES / "batch_learning_rates.png"
)
plt.show()
sgd_alphas = [
    1e-4,
    1e-3,
    1e-2
]
plt.figure(figsize=(8, 5))
for alpha in sgd_alphas:

    theta, history = fit_sgd(
        XA,
        y_train,
        alpha,
        50
    )
    plt.plot(
        history,
        label="alpha = " + str(alpha)
    )
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("SGD Learning Rates")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    FIGURES / "sgd_learning_rates.png"
)
plt.show()
theta_normal_A = fit_normal(
    XA,
    y_train
)
theta_batch_A, batch_history = fit_batch_gd(
    XA,
    y_train,
    1e-4,
    50000
)
theta_sgd_A, sgd_history = fit_sgd(
    XA,
    y_train,
    1e-2,
    500
)
theta_normal_B = fit_normal(
    XB,
    y_train
)
theta_batch_B, batch_history_B = fit_batch_gd(
    XB,
    y_train,
    1e-4,
    50000
)
theta_sgd_B, sgd_history_B = fit_sgd(
    XB,
    y_train,
    1e-2,
    500
)
prediction_normal_A = hypothesis(
    XAt,
    theta_normal_A
)
prediction_batch_A = hypothesis(
    XAt,
    theta_batch_A
)
prediction_sgd_A = hypothesis(
    XAt,
    theta_sgd_A
)
prediction_normal_B = hypothesis(
    XBt,
    theta_normal_B
)
prediction_batch_B = hypothesis(
    XBt,
    theta_batch_B
)
prediction_sgd_B = hypothesis(
    XBt,
    theta_sgd_B
)
prediction_normal_A = np.maximum(
    prediction_normal_A,
    0
)
prediction_batch_A = np.maximum(
    prediction_batch_A,
    0
)
prediction_sgd_A = np.maximum(
    prediction_sgd_A,
    0
)
prediction_normal_B = np.maximum(
    prediction_normal_B,
    0
)
prediction_batch_B = np.maximum(
    prediction_batch_B,
    0
)
prediction_sgd_B = np.maximum(
    prediction_sgd_B,
    0
)
daytime = (
    test["IRRADIATION"].to_numpy() > 0
)
rmse_normal_A_all = rmse(
    y_test,
    prediction_normal_A
)
rmse_normal_A_day = rmse(
    y_test[daytime],
    prediction_normal_A[daytime]
)
rmse_batch_A_all = rmse(
    y_test,
    prediction_batch_A
)
rmse_batch_A_day = rmse(
    y_test[daytime],
    prediction_batch_A[daytime]
)
rmse_sgd_A_all = rmse(
    y_test,
    prediction_sgd_A
)
rmse_sgd_A_day = rmse(
    y_test[daytime],
    prediction_sgd_A[daytime]
)


rmse_normal_B_all = rmse(
    y_test,
    prediction_normal_B
)

rmse_normal_B_day = rmse(
    y_test[daytime],
    prediction_normal_B[daytime]
)


rmse_batch_B_all = rmse(
    y_test,
    prediction_batch_B
)

rmse_batch_B_day = rmse(
    y_test[daytime],
    prediction_batch_B[daytime]
)
rmse_sgd_B_all = rmse(
    y_test,
    prediction_sgd_B
)
rmse_sgd_B_day = rmse(
    y_test[daytime],
    prediction_sgd_B[daytime]
)
print("\n-----------------------------")
print("RMSE RESULTS")
print("-----------------------------")
print("\nSet A - Normal Equation")
print(
    "All hours:",
    rmse_normal_A_all
)
print(
    "Daytime:",
    rmse_normal_A_day
)
print("\nSet A - Batch GD")

print(
    "All hours:",
    rmse_batch_A_all
)
print(
    "Daytime:",
    rmse_batch_A_day
)
print("\nSet A - SGD")

print(
    "All hours:",
    rmse_sgd_A_all
)
print(
    "Daytime:",
    rmse_sgd_A_day
)
print("\nSet B - Normal Equation")

print(
    "All hours:",
    rmse_normal_B_all
)
print(
    "Daytime:",
    rmse_normal_B_day
)


print("\nSet B - Batch GD")
print(
    "All hours:",
    rmse_batch_B_all
)
print(
    "Daytime:",
    rmse_batch_B_day
)
print("\nSet B - SGD")
print(
    "All hours:",
    rmse_sgd_B_all
)

print(
    "Daytime:",
    rmse_sgd_B_day
)
difference_A = np.max(
    np.abs(
        theta_normal_A - theta_batch_A
    )
)

difference_B = np.max(
    np.abs(
        theta_normal_B - theta_batch_B
    )
)
print("\n-----------------------------")
print("THETA DIFFERENCE")
print("-----------------------------")

print("Set A:", difference_A)
print("Set B:", difference_B)
plt.figure(figsize=(10, 5))
plt.plot(
    test["datetime"],
    y_test,
    label="Actual"
)
plt.plot(
    test["datetime"],
    prediction_normal_A,
    label="Set A Prediction"
)
plt.plot(
    test["datetime"],
    prediction_normal_B,
    label="Set B Prediction"
)
plt.xlabel("Datetime")
plt.ylabel("AC Power (kW)")

plt.title(
    "Actual vs Predicted AC Power"
)
plt.legend()
plt.grid(True)

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    FIGURES / "actual_vs_predicted.png"
)

plt.show()
print("\nTask 5.1 - Set A Normal Equation")
print(
    "theta0 (Intercept):",
    theta_normal_A[0]
)
print(
    "theta1 (Irradiation):",
    theta_normal_A[1]
)
print(
    "theta2 (Module Temperature):",
    theta_normal_A[2]
)
print(
    "theta3 (Ambient Temperature):",
    theta_normal_A[3]
)
print(
    "theta4 (Sin Hour):",
    theta_normal_A[4]
)
print(
    "theta5 (Cos Hour):",
    theta_normal_A[5]
)
feature_names = [
    "Irradiation",
    "Module Temperature",
    "Ambient Temperature",
    "Sin Hour",
    "Cos Hour"
]
weights = theta_normal_A[1:]

largest_index = np.argmax(
    np.abs(weights)
)
print("\nLargest weight:")
print(feature_names[largest_index])
print("Weight:", weights[largest_index])
day_rmse_A = rmse_normal_A_day
day_rmse_B = rmse_normal_B_day
difference_kw = (
    day_rmse_B - day_rmse_A
)
peak_power = train[
    "AC_POWER"
].max()
difference_percent = (
    difference_kw / peak_power
) * 100
print("\n-----------------------------")
print("TASK 5.2 - SET B VS SET A")
print("-----------------------------")
print(
    "Set A Daytime RMSE:",
    day_rmse_A,
    "kW"
)
print(
    "Set B Daytime RMSE:",
    day_rmse_B,
    "kW"
)
print(
    "Set B is worse than Set A by:",
    difference_kw,
    "kW"
)
print(
    "Peak hourly AC power:",
    peak_power,
    "kW"
)
print(
    "Difference as % of peak power:",
    difference_percent,
    "%"
)
print("\n-----------------------------")
print("TASK 5.3 - SOLVER COMPARISON")
print("-----------------------------")
print("\nSet A Theta:")
print("Normal Equation:")
print(theta_normal_A)
print("\nBatch Gradient Descent:")
print(theta_batch_A)
print("\nStochastic Gradient Descent:")
print(theta_sgd_A)
normal_batch_difference = np.max(
    np.abs(
        theta_normal_A - theta_batch_A
    )
)
normal_sgd_difference = np.max(
    np.abs(
        theta_normal_A - theta_sgd_A
    )
)
print(
    "\nMaximum difference from Normal Equation:"
)
print(
    "Batch GD:",
    normal_batch_difference
)
print(
    "SGD:",
    normal_sgd_difference
)
print("\nIterations / Epochs:")

print(
    "Normal Equation: 1 direct calculation"
)
print(
    "Batch GD:",
    len(batch_history),
    "iterations"
)
print(
    "SGD:",
    len(sgd_history),
    "epochs"
)
table2 = pd.DataFrame({

    "Solver": [
        "Normal equation",
        "Batch GD",
        "Stochastic GD",
        "Normal equation",
        "Batch GD",
        "Stochastic GD"
    ],
    "Features": [
        "Set A",
        "Set A",
        "Set A",
        "Set B",
        "Set B",
        "Set B"
    ],
    "All hours RMSE (kW)": [
        rmse_normal_A_all,
        rmse_batch_A_all,
        rmse_sgd_A_all,
        rmse_normal_B_all,
        rmse_batch_B_all,
        rmse_sgd_B_all
    ],
    "Daytime only RMSE (kW)": [
        rmse_normal_A_day,
        rmse_batch_A_day,
        rmse_sgd_A_day,
        rmse_normal_B_day,
        rmse_batch_B_day,
        rmse_sgd_B_day
    ]
})
table2.to_csv(
    TABLES / "table2_test_rmse.csv",
    index=False
)
print("\nTable 2 saved:")
print(
    TABLES / "table2_test_rmse.csv"
)
table3 = pd.DataFrame({
    "Feature": [
        "theta0 (intercept)",
        "theta1 (irradiation)",
        "theta2 (module temperature)",
        "theta3 (ambient temperature)",
        "theta4 (sin hour)",
        "theta5 (cos hour)"
    ],
    "Normal equation": theta_normal_A,

    "Batch GD": theta_batch_A,

    "Stochastic GD": theta_sgd_A
})

table3.to_csv(
    TABLES / "table3_learned_theta.csv",
    index=False
)
print("\nTable 3 saved:")
print(
    TABLES / "table3_learned_theta.csv"
)
theta_difference_table = pd.DataFrame({

    "Comparison": [
        "Max |Batch GD - Normal equation|",
        "Max |SGD - Normal equation|"
    ],
    "Value": [
        normal_batch_difference,
        normal_sgd_difference
    ]
})
theta_difference_table.to_csv(
    TABLES / "theta_solver_difference.csv",
    index=False
)
print("\nTheta comparison table saved:")
print(
    TABLES / "theta_solver_difference.csv"
)
plt.figure(figsize=(10, 5))
plt.plot(
    batch_history,
    label="Batch GD"
)
plt.plot(
    sgd_history,
    label="SGD"
)

plt.xlabel("Iteration / Epoch")
plt.ylabel("Cost J(theta)")
plt.title(
    "Batch GD vs SGD Cost Curves"
)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    FIGURES / "batch_vs_sgd_cost.png"
)
plt.show()
residuals = (
    y_test - prediction_normal_A
)
plt.figure(figsize=(10, 5))
plt.scatter(
    test["hour"],
    residuals
)
plt.axhline(
    0,
    linestyle="--"
)
plt.xlabel("Hour of Day")
plt.ylabel(
    "Residual (Actual - Predicted)"
)
plt.title(
    "Residuals vs Hour of Day - Set A Normal Equation"
)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    FIGURES / "residuals_vs_hour.png"
)
plt.show()
mean_B = train[
    features_B
].mean().to_numpy()
std_B = train[
    features_B
].std(
    ddof=0
).to_numpy()

std_B[std_B == 0] = 1
np.save(
    SAVED_WEIGHTS / "theta_normal_B.npy",
    theta_normal_B
)
np.save(
    SAVED_WEIGHTS / "mean_B.npy",
    mean_B
)
np.save(
    SAVED_WEIGHTS / "std_B.npy",
    std_B
)
print("\n================================================")
print("TASK 6 MODEL FILES SAVED")
print("================================================")
print(
    SAVED_WEIGHTS / "theta_normal_B.npy"
)
print(
    SAVED_WEIGHTS / "mean_B.npy"
)
print(
    SAVED_WEIGHTS / "std_B.npy"
)
print("\n================================================")
print("TRAINING AND EVALUATION COMPLETED")
print("================================================")
