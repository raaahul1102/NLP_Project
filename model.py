import os
import tarfile
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Step 1: Download and extract dataset
url = 'https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
data_dir = 'rt-polaritydata'
file_path = 'rt-polaritydata.tar.gz'

# Download the dataset
if not os.path.exists(file_path):
    response = requests.get(url)
    with open(file_path, 'wb') as f:
        f.write(response.content)

# Extract the dataset
if not os.path.exists(data_dir):
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall()

# Step 2: Load and process data
def load_data(file_pos, file_neg):
    with open(file_pos, 'r', encoding='ISO-8859-1') as f_pos:
        pos_data = f_pos.readlines()
    with open(file_neg, 'r', encoding='ISO-8859-1') as f_neg:
        neg_data = f_neg.readlines()
    return pos_data, neg_data

positive_data, negative_data = load_data(f'{data_dir}/rt-polarity.pos', f'{data_dir}/rt-polarity.neg')

# Create labels
labels_pos = [1] * len(positive_data)
labels_neg = [0] * len(negative_data)

# Combine data and labels
data = positive_data + negative_data
labels = labels_pos + labels_neg

# Step 3: Create train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(data[:8000], labels[:8000], test_size=1000, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Step 4: TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Step 5: Train multiple models and evaluate
models = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(n_estimators=100),
}

best_model = None
best_f1 = 0
best_metrics = {}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_val_tfidf)
    
    # Evaluate on validation set
    cm = confusion_matrix(y_val, y_pred)
    print(f"Confusion Matrix for {model_name}:\n{cm}")
    
    # Classification report
    report = classification_report(y_val, y_pred, output_dict=True)
    f1 = report['weighted avg']['f1-score']
    
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_metrics = report

# Step 6: Test best model on the test set
y_test_pred = best_model.predict(X_test_tfidf)
cm_test = confusion_matrix(y_test, y_test_pred)

# Print final confusion matrix
print(f"\nBest Model Confusion Matrix on Test Set:\n{cm_test}")

# Best Model Metrics on the test set
test_report = classification_report(y_test, y_test_pred, output_dict=True)

# Extract precision, recall, f1-score from the test set
precision = test_report['weighted avg']['precision']
recall = test_report['weighted avg']['recall']
f1_score = test_report['weighted avg']['f1-score']

print("\nBest Model Metrics on Test Set:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")

# Optional: print best model and evaluation metrics
print("\nBest Model:", best_model)
print("\nFinal Confusion Matrix (TP, FP, TN, FN):\n", cm_test)
