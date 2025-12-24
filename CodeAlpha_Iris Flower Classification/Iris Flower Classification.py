# Iris Flower Classification
# CodeAlpha Data Science Internship - Task 1

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import kagglehub

# Step 1: Download the dataset from Kaggle
print("Downloading Iris dataset from Kaggle...")
path = kagglehub.dataset_download("saurabh00007/iriscsv")
print("Path to dataset files:", path)

# Step 2: Load the dataset
df = pd.read_csv(path + '/Iris.csv')

print("\n=== Dataset Overview ===")
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nSpecies distribution:\n{df['Species'].value_counts()}")

# Step 3: Data Preprocessing
# Drop the 'Id' column if it exists
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

# Separate features (X) and target (y)
X = df.drop('Species', axis=1)
y = df['Species']

print(f"\n=== Data Preprocessing ===")
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Step 5: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train the K-Nearest Neighbors model
print(f"\n=== Training KNN Model ===")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
print("Model training complete!")

# Step 7: Make predictions
y_pred = knn.predict(X_test_scaled)

# Step 8: Evaluate the model
print(f"\n=== Model Evaluation ===")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Step 9: Example prediction on new data
print("\n=== Example Prediction ===")
example_data = [[5.1, 3.5, 1.4, 0.2]]  # Sample measurements
example_scaled = scaler.transform(example_data)
prediction = knn.predict(example_scaled)
print(f"Input measurements: {example_data[0]}")
print(f"Predicted species: {prediction[0]}")

print("\n=== Task Completed Successfully! ===")