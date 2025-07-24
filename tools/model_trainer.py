# tools/model_trainer.py

from agent.registry import get_data, store_model, get_model
from utils.tool_utils import format_success, format_failure
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import uuid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def train_model_tool(inputs: str) -> dict:
    try:
        X_train = get_data(inputs["X_train_id"])
        y_train = get_data(inputs["y_train_id"])
        if X_train is None or y_train is None:
            return format_failure("Training data not found.")

        model_type = inputs.get("model_type", "logistic_regression").lower()
        params = inputs.get("params", {})

        if model_type == "logistic_regression":
            model = LogisticRegression(**params)
        elif model_type == "random_forest":
            model = RandomForestClassifier(**params)
        elif model_type == "xgboost":
            model = xgb.XGBClassifier(**params)
        else:
            return format_failure(f"Unknown model type: {model_type}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)
        acc = accuracy_score(y_train, y_pred)
        f1 = f1_score(y_train, y_pred)

        model_id = f"{model_type}_{uuid.uuid4().hex[:6]}"
        store_model(model_id, model)

        return format_success(
            outputs={
                "model_id": model_id,
                "metrics": {"train_accuracy": round(acc, 4), "train_f1": round(f1, 4)}
            },
            logs=[f"Model {model_id} trained successfully"]
        )

    except Exception as e:
        return format_failure(str(e))
    


def evaluate_model_tool(inputs: dict) -> dict:
    """
    inputs = {
        "model_id": "random_forest_ab12cd",
        "X_test_id": "X_test",
        "y_test_id": "y_test"
    }
    """
    try:
        model = get_model(inputs["model_id"])
        X_test = get_data(inputs["X_test_id"])
        y_test = get_data(inputs["y_test_id"])

        if model is None:
            return format_failure("Model not found.")
        if X_test is None or y_test is None:
            return format_failure("Test data not found.")

        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4),
        }

        return format_success(
            outputs={"metrics": metrics},
            logs=[f"Evaluation metrics calculated for model {inputs['model_id']}"]
        )

    except Exception as e:
        return format_failure(str(e))

