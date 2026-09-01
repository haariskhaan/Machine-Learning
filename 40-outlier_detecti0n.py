import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

class OutlierHandler:
    def __init__(self, df):
        self.df = df.copy()
        self.results = {}

    def detect_iqr(self, column):
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return self.df[(self.df[column] < lower) | (self.df[column] > upper)]

    def detect_zscore(self, column, threshold=3):
        z_scores = np.abs(stats.zscore(self.df[column]))
        return self.df[z_scores > threshold]

    def handle_winsorize(self, column):
        lower = np.percentile(self.df[column], 1)
        upper = np.percentile(self.df[column], 99)
        self.df[column] = self.df[column].clip(lower=lower, upper=upper)
        return self.df

    def visualize(self, column):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].boxplot(self.df[column])
        axes[0].set_title('Box Plot')
        axes[1].scatter(range(len(self.df)), self.df[column])
        axes[1].set_title('Scatter Plot')
        axes[2].hist(self.df[column], bins=30)
        axes[2].set_title('Histogram')
        plt.tight_layout()
        plt.show()

# Usage Example
df = pd.DataFrame({'Age': [20, 21, 22, 19, 20, 95]})
handler = OutlierHandler(df)
outliers = handler.detect_iqr('Age')
print("Outliers:", outliers)
handler.handle_winsorize('Age')
handler.visualize('Age')
