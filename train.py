# train.py
import os
import json
from pathlib import Path
from urllib.parse import urlparse, unquote

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# -----------------------------
# CONFIG MLflow (tracking+registry)
# -----------------------------
def ensure_tracking_and_registry():
    """
    Configura MLflow para usar un FileStore local (./mlruns).
    Respeta MLFLOW_TRACKING_URI si viene por entorno (p.ej. en Actions).
    """
    mlruns_dir = Path("mlruns")
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    tracking_uri_env = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri_env:
        tracking_uri = tracking_uri_env
    else:
        tracking_uri = mlruns_dir.resolve().as_uri()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    print(f"[DEBUG] Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"[DEBUG] Registry  URI: {mlflow.get_registry_uri()}")
    return tracking_uri


# -----------------------------
# Helpers para comparar URIs/rutas
# -----------------------------
def _uri_to_abs_path(uri_or_path: str) -> Path:
    """
    Convierte 'file:./mlruns', 'file:///C:/...', '/home/runner/...' a Path absoluto normalizado.
    """
    if uri_or_path.startswith("file:"):
        p = urlparse(uri_or_path)
        return Path(unquote(p.path)).resolve()
    return Path(uri_or_path).resolve()


# -----------------------------
# Experimentos (a prueba de Windows/Linux)
# -----------------------------
def get_or_create_experiment(experiment_name: str) -> str:
    """
    Devuelve el experiment_id. Crea el experimento si no existe.
    Si existe pero su artifact_location NO pertenece al store actual (./mlruns del entorno),
    crea/usa un alterno con sufijo '-ci' y artifact_location local al runner.
    """
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    current_store_abs = Path("mlruns").resolve()

    if exp is None:
        experiment_id = client.create_experiment(
            name=experiment_name,
            artifact_location=current_store_abs.as_uri(),
        )
        print(f"[DEBUG] Creado experimento '{experiment_name}' (ID: {experiment_id})")
        mlflow.set_experiment(experiment_name)
        return experiment_id

    # Si existe, validamos compatibilidad de stores comparando rutas reales
    exp_store_abs = _uri_to_abs_path(exp.artifact_location)
    same_root = str(exp_store_abs).startswith(str(current_store_abs))

    if not same_root:
        alt_name = f"{experiment_name}-ci"
        print(
            f"[WARN] artifact_location existente: '{exp.artifact_location}' → {exp_store_abs}\n"
            f"      NO pertenece al store actual ({current_store_abs}). "
            f"Se usará/creará experimento alterno: '{alt_name}'."
        )
        alt = client.get_experiment_by_name(alt_name)
        if alt is None:
            experiment_id = client.create_experiment(
                name=alt_name,
                artifact_location=current_store_abs.as_uri(),
            )
            print(f"[DEBUG] Creado experimento '{alt_name}' (ID: {experiment_id})")
        else:
            experiment_id = alt.experiment_id
            print(f"[DEBUG] Usando experimento existente '{alt_name}' (ID: {experiment_id})")
        mlflow.set_experiment(alt_name)
        return experiment_id

    # Mismo store → usarlo
    print(f"[DEBUG] Usando experimento existente '{experiment_name}' (ID: {exp.experiment_id})")
    print(f"[DEBUG] artifact_location: {exp.artifact_location} → {exp_store_abs}")
    mlflow.set_experiment(experiment_name)
    return exp.experiment_id


# -----------------------------
# Entrenamiento + logging
# -----------------------------
def train_and_log(experiment_id: str, test_size: float = 0.2, seed: int = 42):
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = float(mean_squared_error(y_test, preds))
    print(f"[INFO] MSE (test): {mse:.4f}")

    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        print(f"[DEBUG] run_id: {run_id}")
        print(f"[DEBUG] artifact_uri: {run.info.artifact_uri}")

        mlflow.log_params({
            "model": "LinearRegression",
            "test_size": test_size,
            "random_state": seed,
        })
        mlflow.log_metric("mse", mse)

        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        metrics_file = artifacts_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump({"mse": mse}, f, indent=2)
        mlflow.log_artifact(str(metrics_file))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="models",
        )

    print("✅ Entrenamiento y registro completados.")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    ensure_tracking_and_registry()
    EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "CI-CD-Lab2")
    experiment_id = get_or_create_experiment(EXPERIMENT_NAME)
    train_and_log(experiment_id=experiment_id)

