import streamlit as st
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.tree import DecisionTreeClassifier  # Import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix, roc_auc_score, roc_curve
import numpy as np
from sklearn.naive_bayes import GaussianNB  # Import GaussianNB
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
choice = st.selectbox("Select Model Algo", ["Decision Tree", "Gusion","CART"])
# File uploader

st.title("Decision Tree")   
# Tuning Parameters

with st.expander("Tuning Parameters"):
    test_size = 0.2
    random_seed = 42
    max_depth = 20
    min_samples_split = 10
    min_samples_leaf = 10
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)


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
st.write(f"Mean Classification Accuracy (LOOCV): {mean_accuracy * 100:.2f}%")

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

st.title("Gaussion rapid boots")

var_smoothing=-9
test_size=0.2
random_seed=42
    

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
var_smoothing_value = 10 ** var_smoothing

model = GaussianNB(var_smoothing=var_smoothing_value)
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
st.write(f"Mean Classification Accuracy: {mean_accuracy * 100:.2f}%")
# Visualization

# Save the model
model_filename = "gaussian_nb_model.joblib"
joblib.dump(model, model_filename)

with open(model_filename, "rb") as f:
    model_data = f.read()

st.download_button(
    label="Download Trained Model",
    data=model_data,
    file_name=model_filename,
    mime="application/octet-stream"
)
st.title("Gradient Boosting Classifier with LOOCV")


    # Split features and target variable
   

with st.expander("Tuning Parameters"):
    test_size = 0.3
    random_seed = 42
    n_estimators = 10
    learning_rate = 0.2
    max_depth = 2

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

# Initialize the Gradient Boosting Classifier
model = GradientBoostingClassifier(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    random_state=random_seed
)



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
st.write(f"Mean Classification Accuracy: {mean_accuracy * 100:.2f}%")
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

st.title("K-Nearest Neighbors Classifier with LOOCV")

with st.expander("Tuning Parameters"):
    test_size = st.slider("Test Size (for train-test split)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_seed = st.slider("Random Seed", min_value=0, max_value=100, value=42, step=1)
    n_neighbors = st.slider("Number of Neighbors (k)", min_value=1, max_value=50, value=5, step=1)
    metric = st.selectbox("Distance Metric", options=["euclidean", "manhattan", "chebyshev", "minkowski"], index=0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

st.header("Leave-One-Out Cross-Validation (LOOCV)")
model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
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
st.write(f"Mean Classification Accuracy: {mean_accuracy * 100:.2f}%")
model_filename = "knn_model.joblib"
joblib.dump(model, model_filename)

with open(model_filename, "rb") as f:
    model_data = f.read()

st.download_button(
    label="Download Trained Model",
    data=model_data,
    file_name=model_filename,
    mime="application/octet-stream"
)
with st.expander("Tuning Parameters"):
    test_size = st.slider("Test Size (for train-test split)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_seed = st.slider("Random Seed", min_value=0, max_value=100, value=42, step=1)
    n_neighbors = st.slider("Number of Neighbors (k)", min_value=1, max_value=50, value=5, step=1)
    metric = st.selectbox("Distance Metric", options=["euclidean", "manhattan", "chebyshev", "minkowski"], index=0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

st.header("Leave-One-Out Cross-Validation (LOOCV)")
model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
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
