from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# STEP 1: READ CSV FILE
# ============================================================

print("First 5 rows:")

df = pd.read_csv("Fish-1.csv")

print(df)

print("\nDataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
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
df["Species"] = df["Species"].fillna(df["Species"].mode()[0])

print("\nMissing values after handling:")
print(df.isnull().sum())

print("#######################################")


# ============================================================
# STEP 3: ENCODE SPECIES
# ============================================================

# Species is categorical/string data
# Convert Species into numerical values

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

print("\nData types:")
print(df.dtypes)


print("################################################")


# ============================================================
# STEP 5: FEATURE SCALING
# ============================================================

# Ridge Regression uses regularization.
# Therefore scaling is important.

scaler = StandardScaler()

features = [
    "Weight",
    "Length1",
    "Length2",
    "Length3",
    "Height",
    "Species"
]

df[features] = scaler.fit_transform(df[features])

print("\nData after Standard Scaling:")
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

print("\nTrain Test Split")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ============================================================
# STEP 8: RIDGE REGRESSION
# ============================================================

print("\n############### RIDGE REGRESSION ###############")

# alpha controls regularization strength

ridge_model = Ridge(alpha=1.0)

# Train model

ridge_model.fit(X_train, y_train)


# ============================================================
# STEP 9: MODEL PARAMETERS
# ============================================================

print("\n" + "=" * 50)
print("RIDGE REGRESSION MODEL PARAMETERS")
print("=" * 50)

print("\nAlpha:")
print(ridge_model.alpha)

print("\nCoefficients:")
print(ridge_model.coef_)

print("\nIntercept:")
print(ridge_model.intercept_)


# ============================================================
# STEP 10: PREDICTIONS
# ============================================================

y_pred_ridge = ridge_model.predict(X_test)


# ============================================================
# STEP 11: LINEAR REGRESSION FOR COMPARISON
# ============================================================

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)


# ============================================================
# STEP 12: CALCULATE METRICS
# ============================================================

ridge_mse = mean_squared_error(
    y_test,
    y_pred_ridge
)

ridge_rmse = np.sqrt(ridge_mse)

ridge_mae = mean_absolute_error(
    y_test,
    y_pred_ridge
)

ridge_r2 = r2_score(
    y_test,
    y_pred_ridge
)


# Linear Regression metrics

linear_mse = mean_squared_error(
    y_test,
    y_pred_linear
)

linear_rmse = np.sqrt(linear_mse)

linear_mae = mean_absolute_error(
    y_test,
    y_pred_linear
)

linear_r2 = r2_score(
    y_test,
    y_pred_linear
)


# ============================================================
# STEP 13: MODEL COMPARISON
# ============================================================

print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)


print("\n📉 LINEAR REGRESSION:")

print(f"   MSE:   {linear_mse:.4f}")
print(f"   RMSE:  {linear_rmse:.4f}")
print(f"   MAE:   {linear_mae:.4f}")
print(f"   R²:    {linear_r2:.4f}")


print("\n📊 RIDGE REGRESSION:")

print(f"   MSE:   {ridge_mse:.4f}")
print(f"   RMSE:  {ridge_rmse:.4f}")
print(f"   MAE:   {ridge_mae:.4f}")
print(f"   R²:    {ridge_r2:.4f}")


# ============================================================
# STEP 14: TRAINING METRICS
# ============================================================

y_train_pred_ridge = ridge_model.predict(X_train)

mse_train = mean_squared_error(
    y_train,
    y_train_pred_ridge
)

rmse_train = np.sqrt(mse_train)

mae_train = mean_absolute_error(
    y_train,
    y_train_pred_ridge
)

r2_train = r2_score(
    y_train,
    y_train_pred_ridge
)


# ============================================================
# STEP 15: FINAL RIDGE METRICS
# ============================================================

print("\n" + "=" * 50)
print("RIDGE REGRESSION METRICS")
print("=" * 50)


print("\n📊 Training Set:")

print(f"   MSE:   {mse_train:.4f}")
print(f"   RMSE:  {rmse_train:.4f}")
print(f"   MAE:   {mae_train:.4f}")
print(f"   R²:    {r2_train:.4f}")


print("\n📊 Test Set:")

print(f"   MSE:   {ridge_mse:.4f}")
print(f"   RMSE:  {ridge_rmse:.4f}")
print(f"   MAE:   {ridge_mae:.4f}")
print(f"   R²:    {ridge_r2:.4f}")


# ============================================================
# STEP 16: ACTUAL VS PREDICTED
# ============================================================

print("\n" + "=" * 50)
print("ACTUAL VS PREDICTED")
print("=" * 50)

results = pd.DataFrame({
    "Actual Width": y_test.values,
    "Predicted Width": y_pred_ridge
})

print(results)


# ============================================================
# STEP 17: VISUALIZATIONS
# ============================================================

print("\nStep 17 : Visualizations")


# ------------------------------------------------------------
# Weight vs Width
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Length1 vs Width
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Length2 vs Width
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Length3 vs Width
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Height vs Width
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Species vs Width
# ------------------------------------------------------------

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
# STEP 18: ACTUAL VS PREDICTED PLOT
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred_ridge,
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
    f"Ridge Regression\nR² = {ridge_r2:.4f}"
)

plt.legend()

plt.grid(True, alpha=0.3)

plt.show()


# ============================================================
# STEP 19: RESIDUAL PLOT
# ============================================================

residuals = y_test - y_pred_ridge

plt.figure(figsize=(8, 6))

plt.scatter(
    y_pred_ridge,
    residuals,
    alpha=0.7
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Width")
plt.ylabel("Residual")

plt.title("Ridge Regression - Residual Plot")

plt.grid(True, alpha=0.3)

plt.show()


# ============================================================
# STEP 20: MULTIPLE FEATURE VISUALIZATION
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






plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred_ridge,
    alpha=0.7,
    label="Predicted"
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Width")
plt.ylabel("Predicted Width")

plt.title(
    f"Actual vs Predicted - Ridge Regression\n"
    f"R² = {ridge_r2:.4f}"
)

plt.legend()
plt.grid(True, alpha=0.3)

plt.show()