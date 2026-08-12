from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
# print(df.head())



# to read our csv file
df = pd.read_csv("Fish-1.csv")
print(df)

print(df.isnull().sum())

print("##################################################")


# handling missing values

df["Weight"] = df["Weight"].fillna(df["Weight"].mean())
df["Length1"] = df["Length1"].fillna(df["Length1"].mean())
df["Length2"] = df["Length2"].fillna(df["Length2"].mean())
df["Length3"] = df["Length3"].fillna(df["Length3"].mean())
df["Height"] = df["Height"].fillna(df["Height"].mean())
df["Width"] = df["Width"].fillna(df["Width"].mean())
df["Species"] = df["Species"].fillna(df["Species"].mode()[0])

print(df["Weight"])
print(df["Length1"])
print(df["Length2"])
print(df["Length3"])
print(df["Height"])
print(df["Width"])
print(df["Species"])

# print(df[["Weight", "Length1", "Length2", "Length3", "Height", "Width", "Species"]].isnull().sum())


print("#######################################")
print("#" * 50)

# make object of LabelEncoder()  (Convert String into Int)

le = LabelEncoder()

df["Length1"] = le.fit_transform(df["Length1"])
df["Length2"] = le.fit_transform(df["Length2"])
df["Length3"] = le.fit_transform(df["Length3"])
df["Height"] = le.fit_transform(df["Height"])
df["Width"] = le.fit_transform(df["Width"])
df["Species"] = le.fit_transform(df["Species"])

print(df)

# convert float into integer
df["Weight"] = df["Weight"].astype(int)
df["Species"] = df["Species"].astype(int)
print(df)

print("################################################")

# now use MinMaxScaler tool
scaler = MinMaxScaler(feature_range=(0, 1))
df[["Species"]] = scaler.fit_transform(df[["Species"]])

print(df)

# print("###################### Tran Test Split Function #######################")

X = df[["Weight", "Length1", "Length2", "Length3", "Height", "Species"]]
y = df["Width"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("############### Now Use Polynomial Features ###############")
poly = PolynomialFeatures()

poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)



print("Compare Linear vs Polynomial")

# Train on polynomial features
poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)

# Also train linear model for comparison
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

print("=" * 50)
print("MODEL PARAMETERS (Polynomial Degree 3)")
print("=" * 50)



# Predictions
y_pred_linear = linear_model.predict(X_test)
y_pred_poly = poly_model.predict(X_poly_test)

# Calculate metrics
linear_mse = mean_squared_error(y_test, y_pred_linear)
linear_r2 = r2_score(y_test, y_pred_linear)

poly_mse = mean_squared_error(y_test, y_pred_poly)
poly_r2 = r2_score(y_test, y_pred_poly)

print("=" * 50)
print("MODEL COMPARISON")
print("=" * 50)

print("\n📉 LINEAR REGRESSION:")
print(f"   MSE:  {linear_mse:.2f}")
print(f"   R²:   {linear_r2:.4f}")

print("\n📈 POLYNOMIAL REGRESSION (Degree 3):")
print(f"   MSE:  {poly_mse:.2f}")
print(f"   R²:   {poly_r2:.4f}")

improvement = ((linear_mse - poly_mse) / linear_mse) * 100
print(f"\n✅ MSE Improvement: {improvement:.1f}%")






print("Loss Function & Model Evaluation")

# Test set metrics
mse = mean_squared_error(y_test, y_pred_poly)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_poly)
r2 = r2_score(y_test, y_pred_poly)

# Training metrics (check overfitting)
mse_train = mean_squared_error(y_train, poly_model.predict(X_poly_train))
r2_train = r2_score(y_train, poly_model.predict(X_poly_train))

print("=" * 50)
print("POLYNOMIAL REGRESSION METRICS")
print("=" * 50)

print(f"\n📊 Training Set:")
print(f"   MSE:  {mse_train:.2f}")
print(f"   R²:   {r2_train:.4f}")

print(f"\n📊 Test Set:")
print(f"   MSE (Mean Squared Error):   {mse:.2f}")
print(f"   RMSE (Root MSE):            {rmse:.2f}")
print(f"   MAE (Mean Absolute Error):  {mae:.2f}")
print(f"   R² (Coefficient of Determination): {r2:.4f}")





print("Step 9 : Visualizations")

# Linear vs Polynomial Regression Comparison
plt.figure(figsize=(12, 5))

# Plot 1: Linear Regression
plt.subplot(1, 2, 1)

plt.scatter(df["Weight"], df["Width"])

plt.xlabel("Weight")
plt.ylabel("Width")
plt.title("Weight vs Width")
plt.show()

plt.scatter(df["Length1"], df["Width"])

plt.xlabel("Length1")
plt.ylabel("Width")
plt.title("Length1 vs Width")
plt.show()

plt.scatter(df["Length2"], df["Width"])

plt.xlabel("Length2")
plt.ylabel("Width")
plt.title("Length2 vs Width")
plt.show()

plt.scatter(df["Length3"], df["Width"])

plt.xlabel("Length3")
plt.ylabel("Width")
plt.title("Length3 vs Width")
plt.show()

plt.scatter(df["Height"], df["Width"])

plt.xlabel("Height")
plt.ylabel("Width")
plt.title("Height vs Width")
plt.show()

plt.scatter(df["Species"], df["Width"])
plt.xlabel("Species")
plt.ylabel("Width")
plt.title("Species vs Width")
plt.show()

plt.title(f"Linear Regression (R² = {linear_r2:.4f})")
plt.legend()
plt.tight_layout()
plt.grid(True, alpha=0.3)

print("#######################################################")

features = ["Weight", "Length1", "Length2", "Length3", "Height", "Species"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, feature in zip(axes.ravel(), features):
    ax.scatter(df[feature], df["Width"])
    ax.set_xlabel(feature)
    ax.set_ylabel("Width")
    ax.set_title(f"{feature} vs Width")
    ax.grid(True)

plt.tight_layout()
plt.show()



print("##################################")
print(df.corr()["Width"])
