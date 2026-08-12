import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
# For feature importance
from sklearn.inspection import permutation_importance




df = pd.read_csv("data_poly_multi_2.csv")


# # Correlation matrix
# plt.figure(figsize=(10, 8))
# correlation_matrix = df.corr()
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
# plt.title("Correlation Matrix - All Features")
# plt.show()




# Define features (all except Price)
feature_columns = ['Size', 'Bedrooms', 'Bathrooms', 'Age']
X = df[feature_columns]
y = df['Price']



# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)




# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly_train = poly.fit_transform(X_train_scaled)
X_poly_test = poly.transform(X_test_scaled)


# Train polynomial model
poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)

# Also train simple multiple linear regression for comparison
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)





# Make predictions
y_pred_linear = linear_model.predict(X_test_scaled)
y_pred_poly = poly_model.predict(X_poly_test)

# Calculate metrics
linear_mse = mean_squared_error(y_test, y_pred_linear)
linear_rmse = np.sqrt(linear_mse)
linear_r2 = r2_score(y_test, y_pred_linear)

poly_mse = mean_squared_error(y_test, y_pred_poly)
poly_rmse = np.sqrt(poly_mse)
poly_r2 = r2_score(y_test, y_pred_poly)

print("=" * 60)
print("MODEL COMPARISON - Multiple Features")
print("=" * 60)

print("\n📉 MULTIPLE LINEAR REGRESSION:")
print(f"   MSE:  ${linear_mse:,.0f}")
print(f"   RMSE: ${linear_rmse:,.0f}")
print(f"   R²:   {linear_r2:.4f}")

print("\n📈 MULTIPLE POLYNOMIAL REGRESSION (Degree 2):")
print(f"   MSE:  ${poly_mse:,.0f}")
print(f"   RMSE: ${poly_rmse:,.0f}")
print(f"   R²:   {poly_r2:.4f}")

improvement = ((linear_rmse - poly_rmse) / linear_rmse) * 100
print(f"\n✅ RMSE Improvement: {improvement:.1f}%")





plt.figure(figsize=(10, 6))

plt.scatter(y_test, y_pred_poly, color='green', alpha=0.7, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', linewidth=2, label='Perfect Prediction')

plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.title(f"Predicted vs Actual (Multiple Polynomial Regression)\nR² = {poly_r2:.4f}, RMSE = ${poly_rmse:,.0f}")
plt.legend()
plt.grid(True, alpha=0.3)

# Add metrics on plot
plt.text(0.05, 0.95, f'Features: {len(feature_columns)}\nPolynomial degree: 2',
         transform=plt.gca().transAxes, fontsize=10,
         bbox=dict(facecolor='white', alpha=0.8))

plt.show()