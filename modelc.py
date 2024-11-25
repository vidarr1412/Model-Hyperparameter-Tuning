import streamlit as st
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, roc_curve
import numpy as np

# Add CSS styling
st.markdown("""
    <style>
        .zoom-container {
            display: inline-block;
            transition: transform 0.3s ease-in-out, z-index 0.3s ease-in-out;
            position: relative;
            margin: 10px;
            border-radius: 10px;
            overflow: hidden;
            z-index: 1;
        }
        .zoom-container:hover {
            transform: scale(4.5);
            z-index: 1000;
            position: relative;
        }
        .zoom-container img {
            width: 100%;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Title of the application
st.title("Gradient Boosting Classifier with LOOCV")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    dataframe = pd.read_csv(uploaded_file)

    # Display the dataset
    st.write("Dataset Preview:")
    st.dataframe(dataframe)

    # Split features and target variable
    X = dataframe.drop('DEATH_EVENT', axis=1)  # Features
    y = dataframe['DEATH_EVENT']  # Target

    with st.expander("Tuning Parameters"):
        test_size = st.slider("Test Size (for train-test split)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_seed = st.slider("Random Seed", min_value=0, max_value=100, value=42, step=1)
        n_estimators = st.slider("Number of Estimators", min_value=10, max_value=500, value=100, step=10)
        learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
        max_depth = st.slider("Max Depth of Trees", min_value=1, max_value=10, value=3, step=1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

    # Initialize the Gradient Boosting Classifier
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_seed
    )

    st.header("Leave-One-Out Cross-Validation (LOOCV)")

    # Initialize lists to store results for each iteration
    loocv_accuracies = []
    loocv_log_losses = []
    loocv_probs = []

    # Perform LOOCV
    loocv = LeaveOneOut()
    for train_index, test_index in loocv.split(X):
        X_train_loocv, X_test_loocv = X.iloc[train_index], X.iloc[test_index]
        y_train_loocv, y_test_loocv = y.iloc[train_index], y.iloc[test_index]

        # Fit the model
        model.fit(X_train_loocv, y_train_loocv)

        # Predict
        y_pred = model.predict(X_test_loocv)
        y_prob = model.predict_proba(X_test_loocv)[:, 1]

        # Evaluate
        loocv_accuracies.append(accuracy_score(y_test_loocv, y_pred))
        loocv_log_losses.append(log_loss([y_test_loocv], [y_prob], labels=[0, 1]))
        loocv_probs.extend(y_prob)

    # Calculate mean accuracy and log loss
    mean_accuracy = np.mean(loocv_accuracies)
    mean_log_loss = np.mean(loocv_log_losses)

    # Visualization
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("Classification Accuracy")
        plt.figure(figsize=(10, 5))
        plt.boxplot(loocv_accuracies)
        plt.title('LOOCV Accuracy')
        plt.ylabel('Accuracy')
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        base64_img = base64.b64encode(buffer.read()).decode()
        st.markdown(f'<div class="zoom-container"><img src="data:image/png;base64,{base64_img}" alt="Accuracy Plot"></div>', unsafe_allow_html=True)
        st.write(f"Mean Classification Accuracy: {mean_accuracy * 100:.2f}%")

    with col2:
        st.header("Logarithmic Loss")
        plt.figure(figsize=(10, 5))
        plt.boxplot(loocv_log_losses)
        plt.title('LOOCV Logarithmic Loss')
        plt.ylabel('Logarithmic Loss')
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        base64_img = base64.b64encode(buffer.read()).decode()
        st.markdown(f'<div class="zoom-container"><img src="data:image/png;base64,{base64_img}" alt="Log Loss Plot"></div>', unsafe_allow_html=True)
        st.write(f"Mean Logarithmic Loss: {mean_log_loss:.4f}")

    with col3:
        st.header("Area Under ROC Curve")
        roc_auc = roc_auc_score(y, loocv_probs)
        fpr, tpr, thresholds = roc_curve(y, loocv_probs)

        plt.figure(figsize=(10, 5))
        plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        base64_img = base64.b64encode(buffer.read()).decode()
        st.markdown(f'<div class="zoom-container"><img src="data:image/png;base64,{base64_img}" alt="ROC Curve"></div>', unsafe_allow_html=True)
        st.write(f"Area Under ROC Curve: {roc_auc:.4f}")

    # Save the model
    model_filename = "gradient_boosting_model.joblib"
    joblib.dump(model, model_filename)

    with open(model_filename, "rb") as f:
        model_data = f.read()

    st.download_button(
        label="Download Trained Model",
        data=model_data,
        file_name=model_filename,
        mime="application/octet-stream"
    )
