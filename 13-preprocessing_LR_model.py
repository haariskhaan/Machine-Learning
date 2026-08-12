import pandas as pd

from sklearn.model_selection import train_test_split


from sklearn.linear_model import LinearRegression



df = pd.read_csv("data_LR.csv")

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())



X = df[["Hours"]]   # Feature (independent)
y = df["Marks"]     # Target/Label (dependent)





X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,      # 20% for testing
    random_state=42     # reproducible split
)


# make model
model = LinearRegression()


# train the model on dataset
model.fit(X_train, y_train)



# Predict marks for a student who studied 12 hours
new_hours = [[18]]
predicted_marks = model.predict(new_hours)

print("Predicted marks for 12 hours of study:  ", predicted_marks)

