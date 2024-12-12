# copulse_model_training.py  
# Purpose: This script trains a machine learning model to predict heart disease risk.  
# Steps:  
# 1) Import necessary libraries for data processing and building the model.  
# 2) Load the heart disease dataset from a file.  
# 3) Preprocess the data by cleaning, renaming, and splitting it into features (input) and target (output).  
# 4) Train a Random Forest Classifier (a machine learning model) using the training data.  
# 5) Test the model's accuracy and performance.  
# 6) Save the trained model so it can be used later without retraining.

# Credit: This script is based on and inspired by the work in the repository  
# https://github.com/asthasharma98/Heart-Disease-Prediction-Deployment.git  
# The html input form is also derived from the initial version of this project.

# Import libraries
# These libraries are needed to process data, build models, and evaluate performance
import numpy as np  # For numerical calculations
import pandas as pd  # For handling data in tables
import pickle  # To save and load machine learning models
from sklearn.preprocessing import StandardScaler  # For scaling data to a common range
from sklearn.model_selection import train_test_split  # To split data into training and testing sets
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # To measure model performance
from sklearn.ensemble import RandomForestClassifier  # Random Forest Classifier for prediction
from sklearn.svm import SVC  # Support Vector Machine (not used here but could be an option)
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors (another machine learning model option)
from sklearn.tree import DecisionTreeClassifier  # Decision Tree Classifier (another model option)


# Load the data
# Read the heart disease dataset from a CSV file
heart = pd.read_csv("processed_cleveland.csv")

# Make a copy of the data to avoid accidentally changing the original data
heart_df = heart.copy()

# Rename a column called 'condition' to 'target' for simplicity
# The 'target' column will be our output or goal variable to predict
heart_df = heart_df.rename(columns={'condition': 'target'})
print(heart_df.head())  # Print the first few rows to check the data


# Prepare the data for training and testing
# Separate the features (input data) and the target (output data) for machine learning
x = heart_df.drop(columns='target')  # Features are everything except the target column
y = heart_df.target  # Target is the column we are trying to predict


# Split the data into training and testing sets
# Training data will be used to train the model; testing data will check how well it performs
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


# Scale the features
# Scaling ensures that all features are on the same range, which makes training more accurate
scaler = StandardScaler()  # Create a scaler
x_train_scaler = scaler.fit_transform(x_train)  # Scale the training data
x_test_scaler = scaler.transform(x_test)  # Scale the testing data


# Train the Random Forest model
# Create and train a Random Forest Classifier with 20 decision trees
model = RandomForestClassifier(n_estimators=20)  # Initialize the model
model.fit(x_train_scaler, y_train)  # Train the model using the scaled training data

# Make predictions on the test data
y_pred = model.predict(x_test_scaler)  # Predict results for the testing set
p = model.score(x_test_scaler, y_test)  # Calculate how accurate the model is
print(p)  # Print the model's score


# Evaluate the model
# Check the model's performance using a report and accuracy metrics
print('Classification Report\n', classification_report(y_test, y_pred))  # Print a detailed performance report
print('Accuracy: {}%\n'.format(round((accuracy_score(y_test, y_pred) * 100), 2)))  # Print overall accuracy

# Confusion Matrix: Shows how many predictions matched and how many were incorrect
cm = confusion_matrix(y_test, y_pred)
print(cm)  # Print the confusion matrix


# Save the trained model using pickle
# Pickle allows saving the trained machine learning model to a file
# This way, we don't have to retrain the model every time we want to use it
filename = 'copulse-knnmodel.pkl'  # Name the saved file
pickle.dump(model, open(filename, 'wb'))  # Save the model to a file
