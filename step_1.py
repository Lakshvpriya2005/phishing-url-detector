import pandas as pd

# Load the dataset (replace with your actual file name if different)
df = pd.read_csv('malicious_phish.csv')

# Show the first 5 rows
print(df.head())

# Check basic info
print(df.info())

# Check how many URLs in each class
print(df['type'].value_counts())
from sklearn.preprocessing import LabelEncoder

# Create label encoder object
le = LabelEncoder()

# Encode the 'type' column
df['label'] = le.fit_transform(df['type'])

# Show mapping of labels
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Label Mapping:", label_mapping)

# Check the new column
print(df[['type', 'label']].head())
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Define features and target
X = df['url']
y = df['label']

# Vectorize the URLs
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

print("Vectorization & split completed.")
print("Training data size:", X_train.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Testing data size:", X_test.shape)

# Example new URL to test
test_url = ["http://free-money-wait-now.com"]

# Vectorize the test URL (same vectorizer used for training)
test_vector = vectorizer.transform(test_url)

# Predict
prediction = model.predict(test_vector)

# Map back the label to original type
label_map = {0: 'benign', 1: 'defacement', 2: 'malware', 3: 'phishing'}
print("Predicted class:", label_map[prediction[0]])

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predict for test data
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['benign', 'defacement', 'malware', 'phishing']))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import joblib

# Save the model
joblib.dump(model, 'phishing_url_detector.pkl')

# Save the vectorizer
joblib.dump(vectorizer, 'url_vectorizer.pkl')

print("Model and vectorizer saved successfully!")
