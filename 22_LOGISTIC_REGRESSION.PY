import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
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

# Width ko binary classification mein convert karna
median_width = df['Width'].median()

y = (df['Width'] >= median_width).astype(int)

print("\nClass Distribution:")
print(y.value_counts())

# =========================================================
# 3. Train-Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =========================================================
# 4. Feature Scaling
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# 5. Logistic Regression
# =========================================================

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(X_train_scaled, y_train)

y_pred_logistic = logistic_model.predict(X_test_scaled)

# =========================================================
# 6. Evaluation
# =========================================================

accuracy = accuracy_score(y_test, y_pred_logistic)

print("\n====================================")
print("LOGISTIC REGRESSION")
print("====================================")

print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_logistic))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_logistic))