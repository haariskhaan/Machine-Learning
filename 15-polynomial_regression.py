import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score




df = pd.read_csv("polynomial_data.csv")



# Visualize the non-linear relationship
plt.scatter(df['YearsExperience'], df['Salary'], color='blue', alpha=0.7)
plt.xlabel("Years of Experience")
plt.ylabel("Salary ($)")
plt.title("Experience vs Salary (Non-linear Relationship)")
plt.grid(True, alpha=0.3)
plt.show()




X = df[["YearsExperience"]]  # Feature
y = df["Salary"]             # Target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



# Create polynomial features
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.fit_transform(X_test)




# printing the new features
print("############### Printing New Features poly training set")
print(X_poly_train)
print("############### Printing New Features poly test set")
print(X_poly_test)



print("Original features shape:", X_train.shape)
print("Polynomial features shape:", X_poly_train.shape)
# print("\nFeature names:", poly.get_feature_names_out(['X']))


# Training
# Train on polynomial features
print("############# training polynomial model")
poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)



print("################### Training linear regression model")
# Also train linear model for comparison
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)




# Predictions on X test
y_pred_linear = linear_model.predict(X_test)
y_pred_poly = poly_model.predict(X_poly_test)



# Calculate metrics on linear model
linear_mse = mean_squared_error(y_test, y_pred_linear)
linear_r2 = r2_score(y_test, y_pred_linear)


# Calculate metrics on polynomial model
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




plt.figure(figsize=(12, 5))

# Plot 1: Linear Regression
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', alpha=0.7, label='Actual Data')
plt.plot(X, linear_model.predict(X), color='red', linewidth=2, label='Linear Fit')
plt.xlabel("Years Experience")
plt.ylabel("Salary ($)")
plt.title(f"Linear Regression (R² = {linear_r2:.4f})")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Polynomial Regression
plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue', alpha=0.7, label='Actual Data')

# Sort for smooth curve
X_sorted = np.sort(X, axis=0)
X_poly_sorted = poly.transform(X_sorted)
plt.plot(X_sorted, poly_model.predict(X_poly_sorted),
         color='green', linewidth=2, label='Polynomial Fit (deg 3)')

plt.xlabel("Years Experience")
plt.ylabel("Salary ($)")
plt.title(f"Polynomial Regression (R² = {r2:.4f})")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()






# actual vs predict visualization
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred_poly, color='green', alpha=0.7, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', linewidth=2, label='Perfect Prediction')

plt.xlabel("Actual Salary ($)")
plt.ylabel("Predicted Salary ($)")
plt.title(f"Predicted vs Actual (Polynomial Regression)\nR² = {r2:.4f}")
plt.legend()
plt.grid(True, alpha=0.3)

# Add metrics on plot
plt.text(0.05, 0.95, f'RMSE = ${rmse:,.0f}', transform=plt.gca().transAxes,
         fontsize=11, bbox=dict(facecolor='white', alpha=0.8))

plt.show()