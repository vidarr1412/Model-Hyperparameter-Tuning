import streamlit as st
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.tree import DecisionTreeClassifier  # Import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix, roc_auc_score, roc_curve
import numpy as np
import os

# Add CSS styling
st.markdown("""
                    <style>
                    .zoom-container {
                        display: inline-block;
                        transition: transform 0.3s ease-in-out, z-index 0.3s ease-in-out;
                        position: relative; /* Ensures proper layering */
                        margin: 10px;
                        border-radius: 10px;
                        overflow: hidden;
                        z-index: 1; /* Normal z-index */
                    }

                    .zoom-container:hover {
                        transform: scale(4.5); /* Adjust zoom scale as needed */
                        z-index: 1000; /* Higher value to overlay everything */
                        position: relative; /* Remains positioned properly */
                    }

                    .zoom-container img {
                        width: 100%;
                        border-radius: 10px;
                    }
                    .stColumn {
                        width: 100%;  /* Default width for each column */
                    }
                    .container_body{}
                    .column-1, .column-2, .column-3 {
                        padding: 0px;
                        border-radius: 10px;
                        background-color: #f2f2f2;  /* Light grey background */
                    }

                    .column-1 {
                        background-color: #f2f2f2;  /* Light grey background */
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        margin-top:20px;
                        position:absolute;
                    }

                    .column-2 {
                        background-color: #e0f7fa;  /* Light blue background */
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        margin-top:30px;
                            position:relative;
                    }

                    .column-3 {
                        background-color: #fff3e0;  /* Light orange background */
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        margin-top:40px;
                            position:relative;

                    }
                    .column-4 {
                        background-color: #fff3e0;  /* Light orange background */
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        margin-top:40px;
                            position:relative; margin-left:2000px;
                    }

                    /* Customize dataframe styling */
                    .dataframe th {
                        background-color: #006064; /* Dark blue background for headers */
                        color: white;
                    }
                    .div_1{
                        margin-top:30px;
                        margin-left:200px;
                    }
                    .div_2{
                        margin-top:30px;
                        margin-left:200px;
                    }
                    .grid-container {
                        display: grid;
                        grid-template-columns: 1fr 1fr 1fr;  /* Creates 3 equal columns */
                        gap: 20px;  /* Gap between columns */
                        grid-template-rows: auto auto;  /* 2 rows of auto height */
                    }
                    .column {
                        padding: 20px;
                        border: 1px solid #ccc;
                        border-radius: 8px;
                        background-color: #f9f9f9;
                    }
                    .behind{
                        z-index:0;
                    }
                    </style>
""", unsafe_allow_html=True)

# Title of the application
st.title("Hyper Tuning ML Algorithm")

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

    # Tuning Parameters
    with st.expander("Tuning Parameters"):
        test_size = st.slider("Test Size (for train-test split)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_seed = st.slider("Random Seed", min_value=0, max_value=100, value=42, step=1)
        max_depth = st.slider("Max Depth (Decision Tree)", min_value=1, max_value=20, value=5, step=1)
        min_samples_split = st.slider("Min Samples Split", 2, 10, 2)  # New slider for min_samples_split
        min_samples_leaf = st.slider("Min Samples Leaf", 1, 10, 1)  # New slider for min_samples_leaf

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

    st.header("Leave-One-Out Cross-Validation (LOOCV)")
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,  # Use min_samples_split parameter
        min_samples_leaf=min_samples_leaf,    # Use min_samples_leaf parameter
        random_state=random_seed
    )

    # Initialize lists to store results for each iteration
    loocv_accuracies = []
    loocv_log_losses = []
    loocv_predictions = []
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
        loocv_predictions.extend(y_pred)
        loocv_probs.extend(y_prob)

    # Calculate mean accuracy and log loss
    mean_accuracy = np.mean(loocv_accuracies)
    mean_log_loss = np.mean(loocv_log_losses)

    # Create layout with 3 columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("Classification Accuracy")
        # Boxplot for accuracy
        plt.figure(figsize=(10, 5))
        plt.boxplot(loocv_accuracies)
        plt.title('Leave-One-Out Cross-Validation Accuracy')
        plt.ylabel('Accuracy')
        plt.xticks([1], ['LOO'])
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        base64_img = base64.b64encode(buffer.read()).decode()
        st.markdown(f'<div class="zoom-container"><img src="data:image/png;base64,{base64_img}" alt="Accuracy Plot"></div>', unsafe_allow_html=True)
        st.write(f"Mean Classification Accuracy (LOOCV): {mean_accuracy * 100:.2f}%")

        # Add download button for accuracy plot
        buffer.seek(0)
        st.download_button(
            label="Download Accuracy Plot",
            data=buffer,
            file_name="accuracy_plot.png",
            mime="image/png"
        )

    with col2:
        st.header("Logarithmic Loss")
        # Boxplot for log loss
        plt.figure(figsize=(10, 5))
        plt.boxplot(loocv_log_losses)
        plt.title('Leave-One-Out Cross-Validation Logarithmic Loss')
        plt.ylabel('Logarithmic Loss')
        plt.xticks([1], ['LOO'])
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        base64_img = base64.b64encode(buffer.read()).decode()
        st.markdown(f'<div class="zoom-container"><img src="data:image/png;base64,{base64_img}" alt="Log Loss Plot"></div>', unsafe_allow_html=True)
        st.write(f"Mean Logarithmic Loss (LOOCV): {mean_log_loss:.4f}")

        # Add download button for log loss plot
        buffer.seek(0)
        st.download_button(
            label="Download Log Loss Plot",
            data=buffer,
            file_name="log_loss_plot.png",
            mime="image/png"
        )

    with col3:
        st.header("Area Under ROC Curve")
        # ROC curve
        test_roc_auc = roc_auc_score(y, loocv_probs)
        fpr, tpr, thresholds = roc_curve(y, loocv_probs)

        plt.figure(figsize=(10, 5))
        plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {test_roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        base64_img = base64.b64encode(buffer.read()).decode()
        st.markdown(f'<div class="zoom-container"><img src="data:image/png;base64,{base64_img}" alt="ROC Curve"></div>', unsafe_allow_html=True)
        st.write(f"Area Under ROC Curve: {test_roc_auc:.4f}")

        # Add download button for ROC curve plot
        buffer.seek(0)
        st.download_button(
            label="Download ROC Curve Plot",
            data=buffer,
            file_name="roc_curve_plot.png",
            mime="image/png"
        )

    model_filename = "decision_tree.joblib"
    joblib.dump(model, model_filename)

    # Streamlit download button to download the model
    with open(model_filename, "rb") as f:
        model_data = f.read()

    st.download_button(
        label="Download Trained Model",
        data=model_data,
        file_name=model_filename,
        mime="application/octet-stream"
    )
