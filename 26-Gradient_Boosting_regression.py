import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================================================
# 1. Load Dataset
# =========================================================

df = pd.read_csv("Fish-1.csv")

print("Dataset Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())


# =========================================================
# 3. Features and Label
# =========================================================

X = df[
    ["Weight", "Length1", "Length2", "Length3", "Height"]
]

y = df["Width"]


# =========================================================
# 4. Train-Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# =========================================================
# 5. Feature Scaling
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =========================================================
# 6. Linear Regression
# =========================================================

linear_model = LinearRegression()

linear_model.fit(
    X_train_scaled,
    y_train
)

y_pred_linear = linear_model.predict(
    X_test_scaled
)


# =========================================================
# 7. Gradient Boosting Regression
# =========================================================

gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gb_model.fit(
    X_train_scaled,
    y_train
)

y_pred_gb = gb_model.predict(
    X_test_scaled
)


# =========================================================
# 8. Evaluation Function
# =========================================================

def evaluate_model(model_name, y_test, y_pred):

    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)

    return {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    }


# =========================================================
# 9. Evaluate Models
# =========================================================

results = []

results.append(
    evaluate_model(
        "Linear Regression",
        y_test,
        y_pred_linear
    )
)

results.append(
    evaluate_model(
        "Gradient Boosting",
        y_test,
        y_pred_gb
    )
)


# =========================================================
# 10. Comparison Table
# =========================================================

results_df = pd.DataFrame(results)

print("\n====================================")
print("MODEL COMPARISON")
print("====================================")

print(results_df)


# =========================================================
# 11. Best Model
# =========================================================

best_model = results_df.loc[
    results_df["R2 Score"].idxmax()
]

print("\n====================================")
print("BEST MODEL")
print("====================================")

print(best_model)


# =========================================================
# 12. Actual vs Predicted
# =========================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred_gb,
    alpha=0.7
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--",
    linewidth=2
)

plt.xlabel("Actual Width")
plt.ylabel("Predicted Width")

plt.title(
    f"Actual vs Predicted - Gradient Boosting\n"
    f"R² = {r2_score(y_test, y_pred_gb):.4f}"
)

plt.grid(True, alpha=0.3)

plt.show()


# =========================================================
# 13. Residual Plot
# =========================================================

residuals = y_test - y_pred_gb

plt.figure(figsize=(8, 5))

plt.scatter(
    y_pred_gb,
    residuals,
    alpha=0.7
)

plt.axhline(
    y=0,
    linestyle="--",
    linewidth=2
)

plt.xlabel("Predicted Width")
plt.ylabel("Residuals")

plt.title(
    "Residual Plot - Gradient Boosting Regression"
)

plt.grid(True, alpha=0.3)

mean_residual = np.mean(residuals)

plt.text(
    0.05,
    0.95,
    f"Mean Residual: {mean_residual:.4f}",
    transform=plt.gca().transAxes,
    fontsize=10,
    bbox=dict(
        facecolor="white",
        alpha=0.8
    )
)

plt.show()