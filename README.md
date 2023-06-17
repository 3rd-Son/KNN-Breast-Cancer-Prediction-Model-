<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">

</head>
<body>
  <h1>Breast Cancer Diagnosis Model</h1>

  <h2>Overview</h2>
  <p>This machine learning model predicts whether a cell is malignant or benign based on features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.</p>

  <h2>Dataset</h2>
  <p>The dataset used for training and evaluation is the "Breast Cancer Wisconsin (Diagnostic)" dataset available from the UCI Machine Learning Repository.</p>
  <p>Dataset URL: <a href="https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29">https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29</a></p>
  <p>The dataset contains the following attributes:</p>
  <ol>
    <li>ID number</li>
    <li>Diagnosis (M = malignant, B = benign)</li>
    <li>Ten real-valued features computed for each cell nucleus: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.</li>
    <li>The mean, standard error, and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features.</li>
  </ol>
  <p>The dataset contains a total of 569 instances, with 357 benign and 212 malignant samples.</p>

  <h2>Model Architecture</h2>
  <p>The model uses the k-nearest neighbors (KNN) algorithm for classification. KNN is a non-parametric method that classifies new instances based on their similarity to training instances. In this case, the model calculates the distances to the k nearest neighbors in the feature space and assigns the majority class label among those neighbors.</p>

  <h2>Evaluation</h2>
  <p>The model was trained and evaluated using standard machine learning practices, including train-test split, hyperparameter tuning, and cross-validation. The evaluation metrics used for assessing the model performance include accuracy, precision, recall, and F1-score.</p>
  <p>During evaluation, the model achieved an accuracy of 96% on the test set.</p>

  <h2>Usage</h2>
  <p>To use the trained model for prediction, follow these steps:</p>
  <ol>
    <li>Ensure that you have Python and the required dependencies installed.</li>
    <li>Load the trained model into your Python environment.</li>
    <li>Provide input data with the ten real-valued features for the cell nucleus.</li>
    <li>Call the appropriate method to obtain the predicted diagnosis (malignant or benign).</li>
  </ol>
  <p>Example code:</p>
  <pre><code>import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Load the trained model
model = KNeighborsClassifier(n_neighbors=5)
model.load_model('trained_model.pkl')

# Provide input data
input_data = np.array([[radius, texture, perimeter, area, smoothness, compactness, concavity, concave_points, symmetry, fractal_dimension]])

# Make predictions
predictions = model.predict(input_data)

print(predictions)