import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data_assignment2_matplotlib.csv")

print(type(df))
print(df)




X = df[["Employee", "Department","Experience", "Age"]]
y = df["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(" #################### Training Data: ##################")
print(X_train)

print("Testing Data:")
print(X_test)


