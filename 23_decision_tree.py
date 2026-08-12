import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
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
# 4. Decision Tree Regression
# =========================================================

tree_model = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)

# =========================================================
# 5. Evaluation
# =========================================================

mae = mean_absolute_error(y_test, y_pred_tree)

mse = mean_squared_error(y_test, y_pred_tree)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred_tree)

print("\n====================================")
print("DECISION TREE REGRESSION")
print("====================================")

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

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


results = []

results.append(
    evaluate_model(
        "Decision Tree Regression",
        y_test,
        y_pred_tree
    )
)

results_df = pd.DataFrame(results)

print("\n====================================")
print("MODEL COMPARISON")
print("====================================")

print(results_df)

# =========================================================
# Best Model
# =========================================================

best_model = results_df.loc[
    results_df["R2 Score"].idxmax()
]

print("\n====================================")
print("BEST MODEL")
print("====================================")

print(best_model)