import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# PAGE CONFIG

st.set_page_config(page_title="Churn Prediction App", layout="wide")
st.title("🏦 Bank Customer Churn Prediction")

# LOAD DATA

@st.cache_data
def load_data():
    return pd.read_csv("bank_customer_churn.csv")

data = load_data()

# SHOW DATA

if st.checkbox("Show Raw Data"):
    st.dataframe(data)

# EDA

st.subheader("📊 Churn Distribution")
fig, ax = plt.subplots()
sns.countplot(x='Exited', data=data, ax=ax)
st.pyplot(fig)

# PREPROCESS FUNCTION

def preprocess(data):
    data = data.copy()

    # Remove duplicates
    data.drop_duplicates(inplace=True)

    # Fill missing values
    for col in data.select_dtypes(include=['int64', 'float64']).columns:
        data[col].fillna(data[col].median(), inplace=True)

    for col in data.select_dtypes(include=['object']).columns:
        data[col].fillna(data[col].mode()[0], inplace=True)

    X = data.drop('Exited', axis=1)
    y = data['Exited']

    encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler, encoders

# TRAIN MODELS

def train_and_save_model(data):
    X, y, scaler, encoders = preprocess(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    params = {
        "Logistic Regression": {'C': [0.1, 1]},
        "Decision Tree": {"max_depth":[5,10]},
        "Random Forest": {'n_estimators':[100], 'max_depth':[5,10]},
        "Gradient Boosting": {'n_estimators':[100], 'learning_rate':[0.1]}
    }

    best_model = None
    best_score = 0

    for name in models:
        grid = GridSearchCV(models[name], params[name], cv=3, scoring='f1')
        grid.fit(X_train, y_train)

        model = grid.best_estimator_
        y_pred = model.predict(X_test)
        score = f1_score(y_test, y_pred)

        if score > best_score:
            best_score = score
            best_model = model

    # Save everything
    joblib.dump({
        "model": best_model,
        "scaler": scaler,
        "encoders": encoders,
        "columns": data.drop('Exited', axis=1).columns.tolist()
    }, "best_model.pkl")

    return best_model, scaler, encoders

# LOAD OR TRAIN MODEL

if os.path.exists("best_model.pkl"):
    saved = joblib.load("best_model.pkl")
    model = saved["model"]
    scaler = saved["scaler"]
    encoders = saved["encoders"]
    columns = saved["columns"]

    st.success("✅ Loaded saved model (fast)")
else:
    st.warning("⚠ Training model... please wait")
    model, scaler, encoders = train_and_save_model(data)
    columns = data.drop('Exited', axis=1).columns.tolist()
    st.success("✅ Model trained and saved")

# USER INPUT

st.subheader("🔮 Predict Customer Churn")

input_data = {}

for col in columns:
    if data[col].dtype == 'object':
        input_data[col] = st.selectbox(col, data[col].unique())
    else:
        input_data[col] = st.number_input(
            col,
            float(data[col].min()),
            float(data[col].max()),
            float(data[col].mean())
        )

# PREDICTION

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])

    # Encode
    for col, le in encoders.items():
        input_df[col] = le.transform(input_df[col])

    # Scale
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("❌ Customer is likely to CHURN")
    else:
        st.success("✅ Customer will STAY")

# OPTIONAL: RESET MODEL

if st.button("🔄 Retrain Model"):
    os.remove("best_model.pkl")
    st.warning("Model deleted. Refresh app to retrain.")
