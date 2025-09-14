
from pathlib import Path
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_and_save(model_path: str, C: float = 10.0, gamma: str = "scale") -> dict:
    digits = load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    svm = SVC(C=C, gamma=gamma, kernel="rbf")
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(svm, model_path)
    return {"accuracy": acc, "report": report, "classes": list(range(10))}

def load_model(model_path: str):
    return joblib.load(model_path)
