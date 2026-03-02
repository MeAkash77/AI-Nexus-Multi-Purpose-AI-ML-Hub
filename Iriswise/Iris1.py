import streamlit as st
import os
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# ==============================
# PATH SETUP (CLOUD SAFE)
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Iris.csv")
MODEL_PATH = os.path.join(BASE_DIR, "classifier.pkl")

# ==============================
# LOAD DATASET
# ==============================
dataset = pd.read_csv(DATA_PATH)
dataset.columns = dataset.columns.str.strip()

# Features & Target
x = dataset.drop(["Species", "Id"], axis=1)
y = dataset["Species"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

# ==============================
# LOAD OR TRAIN MODEL
# ==============================
if os.path.exists(MODEL_PATH):
    knn_model = joblib.load(MODEL_PATH)
else:
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(x_train, y_train)
    joblib.dump(knn_model, MODEL_PATH)

# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌸",
    layout="wide"
)

st.title("🌼 Iris Species Prediction App")

# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    st.header("🔧 Settings")

    n_neighbors = st.slider("Neighbors", 1, 15, knn_model.n_neighbors)
    knn_model.n_neighbors = n_neighbors
    knn_model.fit(x_train, y_train)

    show_dataset = st.checkbox("Show Dataset", True)
    show_pairplot = st.checkbox("Show Pairplot", True)
    show_performance = st.checkbox("Show Model Performance", True)
    show_confusion_matrix = st.checkbox("Show Confusion Matrix", True)
    show_model_summary = st.checkbox("Show Model Summary", True)

# ==============================
# INPUT FORM
# ==============================
with st.form("prediction_form"):
    st.subheader("🔍 Enter Flower Features")

    Sepal_length = st.number_input("Sepal Length", 0.0, 10.0, 5.0)
    Sepal_width = st.number_input("Sepal Width", 0.0, 10.0, 3.0)
    Petal_length = st.number_input("Petal Length", 0.0, 10.0, 1.5)
    Petal_width = st.number_input("Petal Width", 0.0, 10.0, 0.2)

    submit_button = st.form_submit_button("🌟 Predict Species")

# ==============================
# PREDICTION
# ==============================
if submit_button:

    x_input = pd.DataFrame(
        [[Sepal_length, Sepal_width, Petal_length, Petal_width]],
        columns=x.columns
    )

    prediction = knn_model.predict(x_input)[0]

    # SAFE IMAGE PATHS
    species_images = {
        "Iris-setosa": os.path.join(BASE_DIR, "assets", "Irissetosa1.jpg"),
        "Iris-versicolor": os.path.join(BASE_DIR, "assets", "Versicolor.webp"),
        "Iris-virginica": os.path.join(BASE_DIR, "assets", "virgina.jpg"),
    }

    st.success(f"🎉 Predicted Species: **{prediction}**")

    img_path = species_images.get(prediction)

    if img_path and os.path.exists(img_path):
        st.image(img_path, caption=f"Iris {prediction}", use_container_width=True)
    else:
        st.warning("Image not found (check assets folder).")

# ==============================
# DATASET VIEW
# ==============================
if show_dataset:
    with st.expander("Dataset Overview"):
        st.write(dataset.head())
        st.bar_chart(dataset["Species"].value_counts())

# ==============================
# PAIRPLOT
# ==============================
if show_pairplot:
    with st.expander("Pairplot"):
        fig = px.scatter_matrix(
            dataset,
            dimensions=x.columns,
            color="Species",
            title="Iris Feature Relationships"
        )
        st.plotly_chart(fig, use_container_width=True)

# ==============================
# PERFORMANCE
# ==============================
if show_performance:
    with st.expander("Model Performance"):

        train_acc = knn_model.score(x_train, y_train) * 100
        test_acc = knn_model.score(x_test, y_test) * 100

        st.write(f"Train Accuracy: {train_acc:.2f}%")
        st.write(f"Test Accuracy: {test_acc:.2f}%")

        if show_confusion_matrix:
            cm = confusion_matrix(
                y_test,
                knn_model.predict(x_test),
                labels=knn_model.classes_
            )

            fig, ax = plt.subplots()
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=knn_model.classes_,
                yticklabels=knn_model.classes_,
                ax=ax,
            )
            st.pyplot(fig)

# ==============================
# MODEL SUMMARY
# ==============================
if show_model_summary:
    with st.expander("Model Summary"):
        st.write(f"Neighbors: {knn_model.n_neighbors}")
        st.write(f"Algorithm: {knn_model._fit_method}")
        st.write(f"Distance Metric: {knn_model.metric}")
