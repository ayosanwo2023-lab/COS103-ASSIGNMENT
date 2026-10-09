import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

iris = pd.read_csv("database/Iris.csv")

features = iris[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
target = iris["Species"]

train_x, test_x, train_y, test_y = train_test_split(
    features, target, test_size=0.2, random_state=42
)

tree = DecisionTreeClassifier(random_state=42)
tree.fit(train_x, train_y)

predictions = tree.predict(test_x)

print("Accuracy:", accuracy_score(test_y, predictions))
print("Precision:", precision_score(test_y, predictions, average="weighted"))
print("Recall:", recall_score(test_y, predictions, average="weighted"))
