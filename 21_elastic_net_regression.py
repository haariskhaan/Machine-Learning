import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================================================
# 1. Load Dataset
# =========================================================

df = pd.read_csv("Fish[1].csv")

print("Dataset Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

# =========================================================
# 2. Features and Label
# =========================================================

X = df[['Weight', 'Length1', 'Length2', 'Length3', 'Height']]
y = df['Width']

# =========================================================
# 3. Train-Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# =========================================================
# 4. Feature Scaling / Preprocessing
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# 5. Linear Regression
# =========================================================

linear_model = LinearRegression()

linear_model.fit(X_train_scaled, y_train)

y_pred_linear = linear_model.predict(X_test_scaled)

# =========================================================
# 6. Polynomial Regression
# =========================================================

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

poly_model = LinearRegression()

poly_model.fit(X_train_poly, y_train)

y_pred_poly = poly_model.predict(X_test_poly)

# =========================================================
# 7. Elastic Net Regression
# =========================================================

elastic_model = ElasticNet(
    alpha=1.0,
    l1_ratio=0.5,
    random_state=42
)

elastic_model.fit(X_train_scaled, y_train)

y_pred_elastic = elastic_model.predict(X_test_scaled)

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
# 9. Evaluate All Models
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
        "Polynomial Regression",
        y_test,
        y_pred_poly
    )
)

results.append(
    evaluate_model(
        "Elastic Net Regression",
        y_test,
        y_pred_elastic
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