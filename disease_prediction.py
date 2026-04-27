import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("dataset.csv")

# Encode disease labels
encoder = LabelEncoder()
df["disease"] = encoder.fit_transform(df["disease"])

# Features and target
X = df.drop("disease", axis=1)
y = df["disease"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# User input
print("\\nEnter Symptoms (0 = No, 1 = Yes)")

fever = int(input("Fever: "))
cough = int(input("Cough: "))
headache = int(input("Headache: "))
fatigue = int(input("Fatigue: "))
vomiting = int(input("Vomiting: "))
body_pain = int(input("Body Pain: "))

symptoms = np.array([
    fever,
    cough,
    headache,
    fatigue,
    vomiting,
    body_pain
]).reshape(1, -1)

prediction = model.predict(symptoms)

result = encoder.inverse_transform(prediction)

print("\\nPredicted Disease:")
print(result[0])
# -----------------------------
# Charts Section
# -----------------------------

import seaborn as sns

# Disease Distribution Chart
plt.figure(figsize=(8, 5))
sns.countplot(x=encoder.inverse_transform(y))
plt.title("Disease Distribution")
plt.xlabel("Disease")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("disease_distribution.png")
plt.show()

# Feature Importance Chart
importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(8, 5))
plt.bar(features, importance)
plt.title("Feature Importance")
plt.xlabel("Symptoms")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# Accuracy Chart
plt.figure(figsize=(6, 5))
plt.bar(["Random Forest"], [accuracy * 100])
plt.title("Model Accuracy")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("accuracy_chart.png")
plt.show()

print("Charts generated successfully!")