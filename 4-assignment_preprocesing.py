from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# to read our csv file
df = pd.read_csv("data.csv")

print(df)

print("\n\n\n##########################################################\n\n\n")

# handling missing values
df["English"] = df["English"].fillna(df["English"].mean())
print(df)

print("################################")

# make object of LabelEncoder()
le = LabelEncoder()

df["Name"] = le.fit_transform(df["Name"])

print(df)


# convert float into integer
df["English"] = df["English"].astype(int)
print(type(df["English"]))
print(df)

print("#########################################################")

# now use MinMaxScaler tool
scaler = MinMaxScaler(feature_range=(0,1))

df[["Physics", "Chemistry"]] = scaler.fit_transform(df[["Physics", "Chemistry"]])

print(df)
