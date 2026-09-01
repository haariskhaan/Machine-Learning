from flask import Flask, render_template, request
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score


app = Flask(__name__)


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("data-2.csv")


# ==========================================
# ENCODE DIAGNOSIS
# ==========================================

le = LabelEncoder()

df["diagnosis"] = le.fit_transform(
    df["diagnosis"]
)


# ==========================================
# FEATURES AND TARGET
# ==========================================

x = df.drop(
    columns=[
        "id",
        "diagnosis",
        "Unnamed: 32"
    ]
)

y = df["diagnosis"]


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# STANDARDIZE FEATURES
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(x_train)

X_test = scaler.transform(x_test)


# ==========================================
# ADABOOST CLASSIFIER
# ==========================================

model = AdaBoostClassifier(
    random_state=42
)

model.fit(
    X_train,
    y_train
)


# ==========================================
# TEST MODEL
# ==========================================

y_pred = model.predict(
    X_test
)


# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
) * 100


print(
    "Accuracy of AdaBoostClassifier is :",
    accuracy_score(y_test, y_pred)
)


# ==========================================
# FEATURE NAMES
# ==========================================

features = list(x.columns)


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        accuracy=round(accuracy, 2),
        features=features
    )


# ==========================================
# PREDICTION
# ==========================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        values = []

        # Get all 30 feature values
        for feature in features:

            value = float(
                request.form[feature]
            )

            values.append(value)


        # ======================================
        # CREATE INPUT DATAFRAME
        # ======================================

        input_data = pd.DataFrame(
            [values],
            columns=features
        )


        # ======================================
        # STANDARDIZE INPUT
        # ======================================

        input_scaled = scaler.transform(
            input_data
        )


        # ======================================
        # PREDICTION
        # ======================================

        prediction = model.predict(
            input_scaled
        )[0]


        # ======================================
        # CONFIDENCE
        # ======================================

        probabilities = model.predict_proba(
            input_scaled
        )[0]


        confidence = round(
            max(probabilities) * 100,
            2
        )


        # ======================================
        # CONVERT BACK TO M / B
        # ======================================

        diagnosis = le.inverse_transform(
            [prediction]
        )[0]


        return render_template(
            "index.html",

            accuracy=round(
                accuracy,
                2
            ),

            features=features,

            prediction=diagnosis,

            confidence=confidence

        )


    except Exception as e:

        return render_template(
            "index.html",

            accuracy=round(
                accuracy,
                2
            ),

            features=features,

            error=str(e)

        )


# ==========================================
# RUN FLASK
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )