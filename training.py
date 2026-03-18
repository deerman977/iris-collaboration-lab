import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from preprocessing import load_data, preprocess_data


def train_model(data):

    X = data.drop("target", axis=1)

    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)

    model.fit(X_train, y_train)

    joblib.dump(model, "iris_model.pkl")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.2f}")

    return model


if __name__ == "__main__":

    data = load_data()

    data = preprocess_data(data)

    train_model(data)