import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv("data_for_multi.csv")

print(df.head())
print(df.info())





features = ["Age", "Experience", "StudyHours", "Advertising"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, feature in zip(axes.ravel(), features):
    ax.scatter(df[feature], df["Sales"])
    ax.set_xlabel(feature)
    ax.set_ylabel("Sales")
    ax.set_title(f"{feature} vs Sales")
    ax.grid(True)

plt.tight_layout()
plt.show()



print("##################################")
print(df.corr()["Sales"])