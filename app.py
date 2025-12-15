import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Global Economy ML Dashboard", layout="wide")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Global Economy Indicators.csv")
    df.columns = df.columns.str.strip()

    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include='object').columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    return df

df = load_data()

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("📌 Dashboard Navigation")
option = st.sidebar.radio(
    "Select Objective",
    [
        "Objective 1: Data Understanding & EDA",
        "Objective 2: Regression",
        "Objective 3: Classification",
        "Objective 4: Clustering",
        "Objective 5: PCA & Neural Network",
        "Objective 6: Ensemble Models"
    ]
)

# -------------------------------------------------
# OBJECTIVE 1
# -------------------------------------------------
if option == "Objective 1: Data Understanding & EDA":
    st.title("📊 Objective 1: Dataset Overview & EDA")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Statistical Summary")
    st.write(df.describe())

    st.subheader("GDP Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Gross Domestic Product (GDP)'], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# -------------------------------------------------
# OBJECTIVE 2
# -------------------------------------------------
elif option == "Objective 2: Regression":
    st.title("📈 Objective 2: Regression Models")

    X = df.drop('Gross Domestic Product (GDP)', axis=1)
    y = df['Gross Domestic Product (GDP)']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.metric("R² Score", round(r2_score(y_test, y_pred), 3))
    st.metric("RMSE", round(np.sqrt(mean_squared_error(y_test, y_pred)), 3))

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Actual GDP")
    ax.set_ylabel("Predicted GDP")
    st.pyplot(fig)

# -------------------------------------------------
# OBJECTIVE 3
# -------------------------------------------------
elif option == "Objective 3: Classification":
    st.title("🧠 Objective 3: Classification")

    df['Economic_Status'] = (df['Per capita GNI'] >= df['Per capita GNI'].median()).astype(int)

    X = df.drop(['Economic_Status', 'Per capita GNI'], axis=1)
    y = df['Economic_Status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.metric("Accuracy", round(acc, 3))

    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", ax=ax)
    st.pyplot(fig)

# -------------------------------------------------
# OBJECTIVE 4
# -------------------------------------------------
elif option == "Objective 4: Clustering":
    st.title("🔍 Objective 4: Clustering")

    features = df[['Population', 'Per capita GNI', 'Gross Domestic Product (GDP)']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    fig, ax = plt.subplots()
    sns.scatterplot(
        x=df['Per capita GNI'],
        y=df['Gross Domestic Product (GDP)'],
        hue=df['Cluster'],
        ax=ax
    )
    st.pyplot(fig)

# -------------------------------------------------
# OBJECTIVE 5
# -------------------------------------------------
elif option == "Objective 5: PCA & Neural Network":
    st.title("📉 Objective 5: PCA & Neural Network")

    X = df.drop('Gross Domestic Product (GDP)', axis=1)
    y = df['Gross Domestic Product (GDP)']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    st.write(f"Reduced Features: {X_pca.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    mlp = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    st.metric("R² Score", round(r2_score(y_test, y_pred), 3))

# -------------------------------------------------
# OBJECTIVE 6
# -------------------------------------------------
elif option == "Objective 6: Ensemble Models":
    st.title("🌲 Objective 6: Ensemble Learning")

    X = df.drop('Gross Domestic Product (GDP)', axis=1)
    y = df['Gross Domestic Product (GDP)']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    st.metric("Random Forest R²", round(r2_score(y_test, y_pred), 3))
