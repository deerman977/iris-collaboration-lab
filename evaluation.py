import pandas as pd
import joblib
import seaborn as sns
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def evaluate_model(model, data):

    X = data.drop("target", axis=1)

    y = data["target"]

    predictions = model.predict(X)

    acc = accuracy_score(y, predictions)

    f1 = f1_score(y, predictions, average='weighted')

    cm = confusion_matrix(y, predictions)

    plt.figure(figsize=(8,6))

    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.savefig("confusion_matrix.png")

    return acc, f1


if __name__ == "__main__":

    data = pd.read_csv("cleaned_data.csv")

    model = joblib.load("iris_model.pkl")

    acc, f1 = evaluate_model(model, data)

    print(f"Accuracy: {acc:.2f}")

    print(f"F1 Score: {f1:.2f}")