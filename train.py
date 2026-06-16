import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset

df = pd.read_csv("datasets.csv")

# Features and target

X = df.drop("fake", axis=1)
y = df["fake"]

# Split data

X_train, X_test, y_train, y_test = train_test_split(
X, y,
test_size=0.2,
random_state=42
)

# Train model

model = RandomForestClassifier(
n_estimators=100,
random_state=42
)

model.fit(X_train, y_train)

# Test accuracy

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# Save model

pickle.dump(model, open("model.pkl", "wb"))

print("Model saved successfully!")
