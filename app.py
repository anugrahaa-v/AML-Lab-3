import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, f_classif
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

st.set_page_config(page_title="Feature Selection Lab", layout="wide")
st.title("MAI511-2 Advanced Machine Learning")
st.subheader("Lab Exercise 3 – Demonstrate Feature Selection")
st.write("Simple interactive demonstration of feature selection.")

diabetes = load_diabetes()
X_reg = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y_reg = pd.Series(diabetes.target)

cancer = load_breast_cancer()
X_clf = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y_clf = pd.Series(cancer.target)

choice = st.sidebar.selectbox("Dataset", ["Diabetes – Linear Regression", "Breast Cancer – Logistic Regression"])
if choice.startswith("Diabetes"):
    X, y, task = X_reg, y_reg, "Regression"
else:
    X, y, task = X_clf, y_clf, "Classification"

st.header("1. Dataset Information")
a,b,c = st.columns(3)
a.metric("Instances", X.shape[0]); b.metric("Original Features", X.shape[1]); c.metric("Task", task)
st.write("Missing values:", int(X.isna().sum().sum()))
with st.expander("View first 10 rows"):
    st.dataframe(X.head(10), use_container_width=True)

if task == "Regression":
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=.2,random_state=42)
else:
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)

scaler = StandardScaler()
Xtr = pd.DataFrame(scaler.fit_transform(Xtr), columns=X.columns)
Xte = pd.DataFrame(scaler.transform(Xte), columns=X.columns)

st.header("2. Feature Selection")
method = st.selectbox("Method", ["Low Variance Filter","High Correlation Filter","Top-K Feature Selection"])

if method == "Low Variance Filter":
    s = VarianceThreshold(.01).fit(Xtr)
    selected = list(X.columns[s.get_support()])
    note = "Variance threshold = 0.01"
elif method == "High Correlation Filter":
    corr = Xtr.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape),1).astype(bool))
    drop = [col for col in upper.columns if any(upper[col] > .80)]
    selected = [col for col in X.columns if col not in drop]
    note = "Correlation threshold = 0.80"
else:
    k = min(5 if task=="Regression" else 10, X.shape[1])
    s = SelectKBest(f_regression if task=="Regression" else f_classif, k=k).fit(Xtr,ytr)
    selected = list(X.columns[s.get_support()])
    note = f"Top {k} features by univariate score"

st.write(note)
st.metric("Retained Features", len(selected))
st.dataframe(pd.DataFrame({"Selected Feature":selected}), use_container_width=True)

fig, ax = plt.subplots()
ax.bar(["Original","Selected"],[X.shape[1],len(selected)])
ax.set_ylabel("Number of Features"); ax.set_title("Feature Count")
st.pyplot(fig)

st.header("3. Model Performance")
if task == "Regression":
    model = LinearRegression().fit(Xtr[selected], ytr)
    pred = model.predict(Xte[selected])
    r2 = r2_score(yte,pred); rmse = mean_squared_error(yte,pred)**.5
    a,b = st.columns(2); a.metric("Test R²",f"{r2:.4f}"); b.metric("Test RMSE",f"{rmse:.4f}")
else:
    model = LogisticRegression(max_iter=5000).fit(Xtr[selected], ytr)
    pred = model.predict(Xte[selected])
    acc = accuracy_score(yte,pred)
    a,b = st.columns(2); a.metric("Test Accuracy",f"{acc:.4f}"); b.metric("Selected Features",len(selected))
    st.dataframe(pd.DataFrame({"Actual":yte.values,"Predicted":pred}).head(20),use_container_width=True)

st.header("4. Lab Notes")
st.markdown("- Low Variance removes features with very little variation.\n"
            "- High Correlation reduces redundant predictors.\n"
            "- Top-K gives a simple interactive feature-selection demonstration.\n"
            "- The complete Colab notebook contains Factor Analysis, Backward Elimination, Forward Selection, "
            "Gradient Descent, k-fold CV and GridSearchCV.")
st.success("Streamlit demonstration ready.")
