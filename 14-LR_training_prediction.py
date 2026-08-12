import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error , r2_score


df = pd.read_csv("data_LR_icecream.csv")

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())



X = df[["Temperature_F"]]   # Feature (independent)
y = df["Ice_Cream_Sales"]     # Target (dependent)





X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,      # 20% for testing
    random_state=42     # reproducible split
)


# make model
model = LinearRegression()


# train the model on dataset
model.fit(X_train, y_train)



y_pred_test = model.predict(X_test)
print(type(y_pred_test))

print("Predictions on test set:", y_pred_test)


# calculate MSE loss function
mse = mean_squared_error(y_test, y_pred_test)
print("MSE   " ,mse)

mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print("MAE   ", mae)
print("R2   ", r2)





# # Predict marks for a student who studied 12 hours
# new_hours = [[18]]
# predicted_marks = model.predict(new_hours)

# print("Predicted marks for 12 hours of study:  ", predicted_marks)

