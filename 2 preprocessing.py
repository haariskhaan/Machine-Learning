import pandas as pd
# to read our csv file
df = pd.read_csv("data.csv")
print(df)

result_column = df["Result"]
print(result_column)

####################################

result_column = df["StudentID"]
print(result_column)

####################################

result_column = df["Name"]
print(result_column)

####################################

result_column = df["Gender"]
print(result_column)

####################################

result_column = df["Math"]
print(result_column)

####################################

result_column = df["Physics"]
print(result_column)

####################################

result_column = df["Chemistry"]
print(result_column)

####################################

result_column = df["English"]
print(result_column)

####################################

result_column = df["Attendance"]
print(result_column)

####################################

result_column = df["Age"]
print(result_column)

print("\n\n\n#########################################\n\n\n")

