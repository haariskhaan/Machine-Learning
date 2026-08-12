from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# read file
df = pd.read_csv("data.csv")

# print data
# print(df)


# handling missing values
df["Age"] = df["Age"].fillna(df["Age"].mean())

# print(df["Age"])

print("######################################")


# removing extra spaces
df["Gender"] = df["Gender"].str.strip()

# convert into upper case
df["Gender"] = df["Gender"].str.capitalize()

print(df["Gender"])




# make object of LabelEncoeder()
le = LabelEncoder()

df["Gender"] = le.fit_transform(df["Gender"])

print(df)

# convert float into integer
df["Age"] = df["Age"].astype(int)
print(type(df["Age"]))
print(df)

print("#########################################################")

# now use MinMaxScaler tool
scaler = MinMaxScaler(feature_range=(-1, 1))

df[["Math", "Physics"]] = scaler.fit_transform(df[["Math", "Physics"]])

print(df)




df.to_csv("E:/Machine-Learning-Course/preprocessed_data.csv")