import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
@st.cache
def load_data():
    return pd.read_csv('cleaned-disease.csv')

data = load_data()

# Extract features and target
@st.cache
def preprocess_data(data):
    X = data.iloc[:, 1:]  # Symptoms (assuming all columns except the first are symptoms)
    y = data.iloc[:, 0]   # Disease (assuming the first column is the target)
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = preprocess_data(data)

# Train and evaluate the selected model
def train_model(model_name):
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "Support Vector Machines (SVM)":
        model = SVC(probability=True)
    elif model_name == "k-Nearest Neighbors (k-NN)":
        model = KNeighborsClassifier(n_neighbors=5)
    else:
        return None, None

    # Train the model
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    # Evaluate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
    
    return model, (accuracy, precision, recall, f1, cm, y_pred, y_proba)

# App title and description
st.title("Disease Prediction App")
st.markdown("""
Select a model, input symptoms, and predict potential diseases. The app also trains the selected model dynamically and displays evaluation metrics.
""")

# Model selection
st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox(
    "Choose a Model",
    ["Logistic Regression", "Random Forest", "Support Vector Machines (SVM)", "k-Nearest Neighbors (k-NN)"]
)

# Train and evaluate the model
if st.sidebar.button("Train Model"):
    with st.spinner("Training the model..."):
        model, metrics = train_model(model_name)
    
    if model is not None and metrics is not None:
        accuracy, precision, recall, f1, cm, y_pred, y_proba = metrics
        
        st.success(f"Model `{model_name}` trained successfully!")
        
        # Display metrics
        st.header("Model Performance Metrics")
        st.metric("Accuracy", f"{accuracy:.2f}")
        st.metric("Precision", f"{precision:.2f}")
        st.metric("Recall", f"{recall:.2f}")
        st.metric("F1 Score", f"{f1:.2f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)
    else:
        st.error("An error occurred while training the model.")

# Input symptoms for prediction
st.header("Input Symptoms")
symptom_list = list(data.columns[1:])  # Assuming symptoms start from the second column
selected_symptoms = st.multiselect("Select symptoms:", symptom_list)

if selected_symptoms:
    # Generate a prediction
    if model is not None:
        # Prepare input data
        input_data = pd.DataFrame([selected_symptoms], columns=X_train.columns).fillna(0)
        prediction = model.predict(input_data)[0]
        confidence = model.predict_proba(input_data).max() if hasattr(model, "predict_proba") else "N/A"
        
        st.header("Prediction Result")
        st.write(f"**Predicted Disease:** {prediction}")
        st.write(f"**Confidence Score:** {confidence:.2f}" if confidence != "N/A" else "**Confidence Score:** Not available")
    else:
        st.warning("Please train a model before making predictions.")

