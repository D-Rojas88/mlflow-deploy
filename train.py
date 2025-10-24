# train.py
import os
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def ensure_tracking_and_registry():
    """
    Configura URIs válidas de MLflow para tracking y registry.
    Por defecto usa almacenamiento local: file:./mlruns
    """
    # Directorio local donde MLflow guardará runs y artefactos
    mlruns_dir = Path("mlruns")
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    # Si el usuario pasó un MLFLOW_TRACKING_URI lo respetamos; si no, usamos file:./mlruns
    tracking_uri_env = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri_env:
        tracking_uri = tracking_uri_env
    else:
        # URI válida multiplataforma (file:///C:/... en Windows, file:///home/... en Linux)
        tracking_uri = mlruns_dir.resolve().as_uri()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    print(f"[DEBUG] Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"[DEBUG] Registry  URI: {mlflow.get_registry_uri()}")

    return tracking_uri


def get_or_create_experiment(experiment_name: str) -> str:
    """
    Devuelve el experiment_id. Si el experimento no existe, lo crea
    con artifact_location en ./mlruns (mismo FileStore).
    """
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        artifact_location = Path("mlruns").resolve().as_uri()
        experiment_id = client.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
        )
        print(f"[DEBUG] Creado experimento '{experiment_name}' (ID: {experiment_id})")
    else:
        experiment_id = exp.experiment_id
        print(f"[DEBUG] Usando experimento existente '{experiment_name}' (ID: {experiment_id})")
        print(f"[DEBUG] artifact_location: {exp.artifact_location}")
    # También setea el contexto por comodidad (no es estrictamente necesario)
    mlflow.set_experiment(experiment_name)
    return experiment_id


def train_and_log(experiment_id: str, test_size: float = 0.2, seed: int = 42):
    """
    Entrena un modelo simple y registra todo en MLflow.
    """
    # Datos
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = float(mean_squared_error(y_test, preds))

    print(f"[INFO] MSE (test): {mse:.4f}")

    # Run de MLflow
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        print(f"[DEBUG] run_id: {run_id}")
        print(f"[DEBUG] artifact_uri: {run.info.artifact_uri}")

        # Parámetros y métricas
        mlflow.log_params({
            "model": "LinearRegression",
            "test_size": test_size,
            "random_state": seed,
        })
        mlflow.log_metric("mse", mse)

        # Artefacto auxiliar (resumen)
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        with open(artifacts_dir / "metrics.json", "w") as f:
            json.dump({"mse": mse}, f, indent=2)
        mlflow.log_artifact(str(artifacts_dir / "metrics.json"))

        # Registrar modelo (sklearn flavor). Quedará en:
        # mlruns/<exp_id>/<run_id>/artifacts/models/...
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="models",
        )

    print("✅ Entrenamiento y registro completados.")


if __name__ == "__main__":
    # 1) Configurar MLflow (tracking/registry locales)
    ensure_tracking_and_registry()

    # 2) Obtener/crear experimento
    EXPERIMENT_NAME = "CI-CD-Lab2"
    experiment_id = get_or_create_experiment(EXPERIMENT_NAME)

    # 3) Entrenar y registrar
    train_and_log(experiment_id=experiment_id)
