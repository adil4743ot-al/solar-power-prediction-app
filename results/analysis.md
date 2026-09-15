# Task 5 – Analysis

## 5.1 Learned Weights for Set A

| Parameter | Normal Equation | Batch GD | Stochastic GD |
|---|---:|---:|---:|
| theta0 (intercept) | 6883.4166 | 6883.4166 | 6877.8543 |
| theta1 (irradiation) | 8258.2965 | 8258.2964 | 8179.3379 |
| theta2 (module temperature) | -25.5450 | -25.5448 | 92.0900 |
| theta3 (ambient temperature) | -15.9294 | -15.9294 | 0.3295 |
| theta4 (sin hour) | -29.8326 | -29.8326 | -42.5369 |
| theta5 (cos hour) | -408.7725 | -408.7725 | -398.5991 |

The largest absolute feature weight is the irradiation coefficient, **8258.2965**.

The positive irradiation coefficient is physically expected because greater solar irradiation generally produces greater PV power. The temperature coefficients are much smaller. The sine and cosine terms represent the daily time-of-day pattern.

## 5.2 Set B Compared with Set A

Daytime test RMSE:
- Set A: **723.4147 kW**
- Set B: **3417.6468 kW**

Difference: **2694.2320 kW**

Training peak AC power: **27325.8987 kW**

The RMSE difference is approximately **9.8596% of the training peak AC power**.

Set B performs substantially worse than Set A. From a rooftop installer perspective, public weather data can still be useful when on-site sensors are unavailable, but it should not be considered an equally accurate replacement for plant-specific measurements for this dataset.

## 5.3 Comparison of the Three Solvers

The three solvers are:
- Normal Equation
- Batch Gradient Descent
- Stochastic Gradient Descent

Settings:
- Normal Equation: direct solution
- Batch GD: alpha = 0.0001, 50000 iterations
- SGD: alpha = 0.01, 500 epochs

For Set A, the maximum absolute difference between Batch GD and the Normal Equation parameters is approximately **0.0001425**.

For Set B, the corresponding difference is approximately **2.91 × 10^-11**.

Therefore, Batch GD converged essentially to the same solution as the Normal Equation.

SGD has similar prediction performance, but its parameters are less close to the direct solution because it updates the model using individual observations and therefore has stochastic variation.

For this small dataset, the Normal Equation is convenient and direct. For a dataset of around 10 million rows, SGD or another iterative approach is more practical because it can process observations incrementally and avoid a large full-matrix calculation.

## 5.4 Batch GD versus SGD Cost Curves

Batch Gradient Descent calculates each update using the complete training dataset. Its cost curve is therefore generally smoother.

SGD updates the parameters one observation at a time. Its cost curve is expected to be noisier and can fluctuate while following an overall downward trend.

The generated learning-rate figures should be used to show these differences.

## 5.5 Residuals versus Hour

Residual is defined as:

**residual = actual AC power − predicted AC power**

The largest hourly residual RMSE values are:

| Hour | Residual RMSE (kW) |
|---:|---:|
| 10 | 1214.74 |
| 14 | 1161.63 |
| 13 | 946.97 |
| 11 | 941.37 |
| 9 | 862.91 |
| 15 | 775.13 |
| 12 | 707.41 |
| 8 | 490.13 |

The largest errors occur mainly during daylight hours, especially around 09:00–15:00.

Possible physical causes include nonlinear PV behaviour, changes in cloud conditions, panel heating, irradiation variability, and inverter/system effects that are not fully represented by the simple linear regression model.

## Overall Conclusion

Set A performs much better than Set B for this solar plant dataset because the on-site irradiation and temperature measurements are more directly related to actual plant conditions.

Irradiation is the dominant learned feature in Set A. The close agreement between Batch GD and the Normal Equation supports the correctness of the gradient-descent implementation.

For the current small dataset, the Normal Equation is simple and effective. For very large datasets, SGD becomes more attractive because it can process observations incrementally.
