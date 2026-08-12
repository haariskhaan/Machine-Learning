# ============================================================
# LASSO REGRESSION - FISH DATASET
# ============================================================

from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LinearRegression

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ============================================================
# STEP 1: READ CSV FILE
# ============================================================

print("\nFirst 5 rows:")

df = pd.read_csv("Fish-1.csv")

print(df)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("##################################################")


# ============================================================
# STEP 2: HANDLING MISSING VALUES
# ============================================================

df["Weight"] = df["Weight"].fillna(df["Weight"].mean())
df["Length1"] = df["Length1"].fillna(df["Length1"].mean())
df["Length2"] = df["Length2"].fillna(df["Length2"].mean())
df["Length3"] = df["Length3"].fillna(df["Length3"].mean())
df["Height"] = df["Height"].fillna(df["Height"].mean())
df["Width"] = df["Width"].fillna(df["Width"].mean())
df["Species"] = df["Species"].fillna(
    df["Species"].mode()[0]
)

print("\nMissing Values After Handling:")
print(df.isnull().sum())

print("#######################################")


# ============================================================
# STEP 3: ENCODE SPECIES
# ============================================================

# Species is categorical data
# Convert String into Integer

le = LabelEncoder()

df["Species"] = le.fit_transform(df["Species"])

print("\nSpecies after Label Encoding:")
print(df["Species"].head())

print("\nSpecies Mapping:")

for i, species in enumerate(le.classes_):
    print(species, "=", i)

print("#######################################")


# ============================================================
# STEP 4: CONVERT DATA TYPES
# ============================================================

df["Weight"] = df["Weight"].astype(float)
df["Length1"] = df["Length1"].astype(float)
df["Length2"] = df["Length2"].astype(float)
df["Length3"] = df["Length3"].astype(float)
df["Height"] = df["Height"].astype(float)
df["Width"] = df["Width"].astype(float)
df["Species"] = df["Species"].astype(int)

print("\nData Types:")
print(df.dtypes)

print("################################################")


# ============================================================
# STEP 5: FEATURE SCALING
# ============================================================

# Scaling is important for Lasso Regression
# because Lasso uses regularization.

scaler = StandardScaler()

features = [
    "Weight",
    "Length1",
    "Length2",
    "Length3",
    "Height",
    "Species"
]

df[features] = scaler.fit_transform(
    df[features]
)

print("\nData After Standard Scaling:")
print(df.head())

print("################################################")


# ============================================================
# STEP 6: DEFINE X AND y
# ============================================================

X = df[
    [
        "Weight",
        "Length1",
        "Length2",
        "Length3",
        "Height",
        "Species"
    ]
]

y = df["Width"]

print("\nX:")
print(X.head())

print("\ny:")
print(y.head())


# ============================================================
# STEP 7: TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n############### TRAIN TEST SPLIT ###############")

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ============================================================
# STEP 8: LASSO REGRESSION
# ============================================================

print("\n############### LASSO REGRESSION ###############")

# alpha controls regularization strength

lasso_model = Lasso(
    alpha=0.01
)

# Train model

lasso_model.fit(
    X_train,
    y_train
)

print("\nLasso model trained successfully!")


# ============================================================
# STEP 9: MODEL PARAMETERS
# ============================================================

print("\n" + "=" * 50)
print("LASSO REGRESSION MODEL PARAMETERS")
print("=" * 50)

print("\nAlpha:")
print(lasso_model.alpha)

print("\nCoefficients:")
print(lasso_model.coef_)

print("\nIntercept:")
print(lasso_model.intercept_)


# ============================================================
# STEP 10: PREDICTIONS
# ============================================================

y_pred_lasso = lasso_model.predict(
    X_test
)


# ============================================================
# STEP 11: LINEAR REGRESSION FOR COMPARISON
# ============================================================

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

y_pred_linear = linear_model.predict(
    X_test
)


# ============================================================
# STEP 12: LASSO METRICS
# ============================================================

lasso_mse = mean_squared_error(
    y_test,
    y_pred_lasso
)

lasso_rmse = np.sqrt(
    lasso_mse
)

lasso_mae = mean_absolute_error(
    y_test,
    y_pred_lasso
)

lasso_r2 = r2_score(
    y_test,
    y_pred_lasso
)


# ============================================================
# STEP 13: LINEAR REGRESSION METRICS
# ============================================================

linear_mse = mean_squared_error(
    y_test,
    y_pred_linear
)

linear_rmse = np.sqrt(
    linear_mse
)

linear_mae = mean_absolute_error(
    y_test,
    y_pred_linear
)

linear_r2 = r2_score(
    y_test,
    y_pred_linear
)


# ============================================================
# STEP 14: MODEL COMPARISON
# ============================================================

print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)


print("\n📉 LINEAR REGRESSION:")

print(f"   MSE:   {linear_mse:.4f}")
print(f"   RMSE:  {linear_rmse:.4f}")
print(f"   MAE:   {linear_mae:.4f}")
print(f"   R²:    {linear_r2:.4f}")


print("\n📊 LASSO REGRESSION:")

print(f"   MSE:   {lasso_mse:.4f}")
print(f"   RMSE:  {lasso_rmse:.4f}")
print(f"   MAE:   {lasso_mae:.4f}")
print(f"   R²:    {lasso_r2:.4f}")


# ============================================================
# STEP 15: TRAINING METRICS
# ============================================================

y_train_pred_lasso = lasso_model.predict(
    X_train
)

mse_train = mean_squared_error(
    y_train,
    y_train_pred_lasso
)

rmse_train = np.sqrt(
    mse_train
)

mae_train = mean_absolute_error(
    y_train,
    y_train_pred_lasso
)

r2_train = r2_score(
    y_train,
    y_train_pred_lasso
)


# ============================================================
# STEP 16: FINAL LASSO REGRESSION METRICS
# ============================================================

print("\n" + "=" * 50)
print("LASSO REGRESSION METRICS")
print("=" * 50)


print("\n📊 Training Set:")

print(f"   MSE:   {mse_train:.4f}")
print(f"   RMSE:  {rmse_train:.4f}")
print(f"   MAE:   {mae_train:.4f}")
print(f"   R²:    {r2_train:.4f}")


print("\n📊 Test Set:")

print(f"   MSE:   {lasso_mse:.4f}")
print(f"   RMSE:  {lasso_rmse:.4f}")
print(f"   MAE:   {lasso_mae:.4f}")
print(f"   R²:    {lasso_r2:.4f}")


# ============================================================
# STEP 17: ACTUAL VS PREDICTED VALUES
# ============================================================

print("\n" + "=" * 50)
print("ACTUAL VS PREDICTED VALUES")
print("=" * 50)

results = pd.DataFrame({
    "Actual Width": y_test.values,
    "Predicted Width": y_pred_lasso
})

# Calculate error

results["Error"] = (
    results["Actual Width"]
    - results["Predicted Width"]
)

print(results)


# ============================================================
# STEP 18: ACTUAL VS PREDICTED PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    range(len(y_test)),
    y_test.values,
    marker="o",
    label="Actual Values"
)

plt.plot(
    range(len(y_test)),
    y_pred_lasso,
    marker="x",
    label="Predicted Values"
)

plt.xlabel("Test Sample")
plt.ylabel("Width")

plt.title(
    "Lasso Regression - Actual vs Predicted"
)

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.show()


# ============================================================
# STEP 19: ACTUAL VS PREDICTED SCATTER PLOT
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred_lasso,
    alpha=0.7,
    label="Predicted"
)

# Perfect prediction line

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Width")
plt.ylabel("Predicted Width")

plt.title(
    f"Actual vs Predicted - Lasso Regression\n"
    f"R² = {lasso_r2:.4f}"
)

plt.legend()

plt.grid(True, alpha=0.3)

plt.show()


# ============================================================
# STEP 20: RESIDUAL PLOT
# ============================================================

residuals = y_test - y_pred_lasso

plt.figure(figsize=(8, 5))

plt.scatter(
    y_pred_lasso,
    residuals,
    color="purple",
    alpha=0.7
)

plt.axhline(
    y=0,
    color="red",
    linestyle="--",
    linewidth=2
)

plt.xlabel("Predicted Width")
plt.ylabel("Residuals")

plt.title(
    "Residual Plot - Lasso Regression"
)

plt.grid(True, alpha=0.3)


# Mean residual

mean_residual = np.mean(
    residuals
)

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


# ============================================================
# STEP 21: WEIGHT VS WIDTH
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["Weight"],
    df["Width"],
    alpha=0.7
)

plt.xlabel("Weight")
plt.ylabel("Width")

plt.title("Weight vs Width")

plt.grid(True, alpha=0.3)

plt.show()


# ============================================================
# STEP 22: LENGTH1 VS WIDTH
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["Length1"],
    df["Width"],
    alpha=0.7
)

plt.xlabel("Length1")
plt.ylabel("Width")

plt.title("Length1 vs Width")

plt.grid(True, alpha=0.3)

plt.show()


# ============================================================
# STEP 23: LENGTH2 VS WIDTH
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["Length2"],
    df["Width"],
    alpha=0.7
)

plt.xlabel("Length2")
plt.ylabel("Width")

plt.title("Length2 vs Width")

plt.grid(True, alpha=0.3)

plt.show()


# ============================================================
# STEP 24: LENGTH3 VS WIDTH
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["Length3"],
    df["Width"],
    alpha=0.7
)

plt.xlabel("Length3")
plt.ylabel("Width")

plt.title("Length3 vs Width")

plt.grid(True, alpha=0.3)

plt.show()


# ============================================================
# STEP 25: HEIGHT VS WIDTH
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["Height"],
    df["Width"],
    alpha=0.7
)

plt.xlabel("Height")
plt.ylabel("Width")

plt.title("Height vs Width")

plt.grid(True, alpha=0.3)

plt.show()


# ============================================================
# STEP 26: SPECIES VS WIDTH
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    df["Species"],
    df["Width"],
    alpha=0.7
)

plt.xlabel("Species")
plt.ylabel("Width")

plt.title("Species vs Width")

plt.grid(True, alpha=0.3)

plt.show()


# ============================================================
# STEP 27: MULTIPLE FEATURE VISUALIZATION
# ============================================================

features = [
    "Weight",
    "Length1",
    "Length2",
    "Length3",
    "Height",
    "Species"
]

fig, axes = plt.subplots(
    2,
    3,
    figsize=(14, 8)
)

for ax, feature in zip(
    axes.ravel(),
    features
):

    ax.scatter(
        df[feature],
        df["Width"],
        alpha=0.7
    )

    ax.set_xlabel(feature)
    ax.set_ylabel("Width")

    ax.set_title(
        f"{feature} vs Width"
    )

    ax.grid(True, alpha=0.3)


plt.tight_layout()

plt.show()


# ============================================================
# STEP 28: LASSO COEFFICIENTS
# ============================================================

print("\n" + "=" * 50)
print("LASSO COEFFICIENTS")
print("=" * 50)

for feature, coefficient in zip(
    features,
    lasso_model.coef_
):

    print(
        f"{feature}: {coefficient:.6f}"
    )