import pandas as pd
# to read our csv file
df = pd.read_csv("data.csv")
print(df)


print("\n\n\n##########################################################\n\n\n")


df["Age"] = df["Age"].fillna(df["Age"].mean())
print(df)

print("\n\n\n##########################################################\n\n\n")

df["Age"] = df["Age"].fillna(df["Age"].median())
print(df)

print("\n\n\n##########################################################\n\n\n")

df["Age"] = df["Age"].fillna(df["Age"].mode()[0])
print(df)