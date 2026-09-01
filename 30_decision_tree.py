import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("data-2.csv")

# print(df.head())
#
# print(df.isnull().sum())

le = LabelEncoder()

df["diagnosis"]=le.fit_transform(df["diagnosis"])

# print(df["diagnosis"])
x = df.drop(columns=["id","diagnosis","Unnamed: 32"])


y = df["diagnosis"]


# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)


# Standardize features
scaler = StandardScaler()

X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)


model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy of DecisionTreeClassifier is :", accuracy_score(y_test, y_pred))

