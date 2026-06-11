import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = make_classification(
    n_samples=500,
    n_features=5,
    n_informative=3,
    random_state=42
)

df = pd.DataFrame(X, columns=[
    "monthly_charges",
    "tenure",
    "support_calls",
    "contract_length",
    "usage_score"
])

df["churn"] = y

X = df.drop("churn", axis=1)
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Customer Churn Prediction Project")
print("Model Accuracy:", round(accuracy * 100, 2), "%")