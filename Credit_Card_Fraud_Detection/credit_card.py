import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)


# Load Datasets
train = pd.read_csv("fraudTrain.csv")
test = pd.read_csv("fraudTest.csv")

print("Training Dataset Shape:", train.shape)
print("Testing Dataset Shape:", test.shape)

# Drop unnecessary columns

drop_cols = ["Unnamed: 0", "trans_date_trans_time",
             "cc_num", "first", "last",
             "street", "city", "state",
             "zip", "dob", "trans_num"]

for col in drop_cols:
    if col in train.columns:
        train.drop(col, axis=1, inplace=True)
        test.drop(col, axis=1, inplace=True)
# Encode categorical columns

from sklearn.preprocessing import LabelEncoder

for col in train.columns:
    if train[col].dtype == "object":

        encoder = LabelEncoder()

        # Fit on both train and test values together
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)

        encoder.fit(combined)

        train[col] = encoder.transform(train[col].astype(str))
        test[col] = encoder.transform(test[col].astype(str))
        
# Features and Target

X_train = train.drop("is_fraud", axis=1)
y_train = train["is_fraud"]

X_test = test.drop("is_fraud", axis=1)
y_test = test["is_fraud"]

# Train Model

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(round(accuracy * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()

plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png", dpi=300)

plt.show()

# ROC Curve

y_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))

plt.plot(fpr, tpr, label="AUC = %.2f" % roc_auc)

plt.plot([0,1],[0,1],'r--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.savefig("roc_curve.png", dpi=300)

plt.show()

# Sample Prediction

sample = X_test.iloc[[0]]

prediction = model.predict(sample)

print("\nSample Prediction:")

if prediction[0] == 1:
    print("Fraudulent Transaction")
else:
    print("Legitimate Transaction")
