# ---------------------------------------------
# Spam SMS Detection using Machine Learning
# Internship Project - CODSOFT
# ---------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# ----------------------------
# Load Dataset
# ----------------------------

df = pd.read_csv("spam.csv", encoding='latin-1')

# Remove unnecessary columns
df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])

# Rename columns
df.columns = ["label", "message"]

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

# ----------------------------
# Convert Labels
# ham = 0
# spam = 1
# ----------------------------

df["label"] = df["label"].map({"ham": 0, "spam": 1})

# ----------------------------
# Features and Target
# ----------------------------

X = df["message"]
y = df["label"]

# ----------------------------
# Convert Text to Numbers
# ----------------------------

vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(X)

# ----------------------------
# Split Dataset
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------
# Train Model
# ----------------------------

model = MultinomialNB()

model.fit(X_train, y_train)

# ----------------------------
# Predictions
# ----------------------------

y_pred = model.predict(X_test)

# ----------------------------
# Accuracy
# ----------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(round(accuracy * 100, 2), "%")

# ----------------------------
# Classification Report
# ----------------------------

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ----------------------------
# Confusion Matrix
# ----------------------------

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()

plt.title("Confusion Matrix")
plt.show()

# ----------------------------
# ROC Curve
# ----------------------------

y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))

plt.plot(fpr, tpr, label="AUC = %0.2f" % roc_auc)

plt.plot([0,1],[0,1],'r--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend(loc="lower right")

plt.show()

# ----------------------------
# Interactive Prediction
# ----------------------------

print("\n----------- Spam SMS Prediction -----------")

while True:

    sms = input("\nEnter an SMS (or type exit): ")

    if sms.lower() == "exit":
        print("Program Ended.")
        break

    sms_vector = vectorizer.transform([sms])

    prediction = model.predict(sms_vector)

    if prediction[0] == 1:
        print("Prediction: SPAM")
    else:
        print("Prediction: HAM (Not Spam)")
