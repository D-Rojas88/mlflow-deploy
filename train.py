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


# ------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ------------------------------------------------------

def ensure_tracking_and_registry():
    """
    Configura URIs válidas de MLflow para tracking y registry.
    Por defecto usa almacenamiento local: file:./mlruns
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


# ------------------------------------------------------
# CREAR O RECUPERAR EXPERIMENTO (COMPATIBLE WINDOWS/LINUX)
# ------------------------------------------------------

def get_or_create_experiment(experiment_name: str) -> str:
    """
    Devuelve el experiment_id. Si el experimento ya existe pero
    tiene un artifact_location incompatible con el entorno actual,
    crea uno nuevo con sufijo '-ci'.
    """
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    current_store = Path("mlruns").resolve().as_uri()

    if exp is None:
        experiment_id = client.create_experiment(
            name=experiment_name,
            artifact_location=current_store,
        )
        print(f"[DEBUG] Creado experimento '{experiment_name}' (ID: {experiment_id})")
        mlflow.set_experiment(experiment_name)
        return experiment_id

    # Si el experimento ya existe, comprobamos si el artifact_location coincide
    if not str(exp.artifact_location).startswith(current_store):
        alt_name = f"{experiment_name}-ci"
        print(f"[WARN] artifact_location del experimento existente es '{exp.artifact_location}' "
              f"≠ store actual '{current_store}'. Creando experimento alterno: '{alt_name}'.")
        alt = client.get_experiment_by_name(alt_name)
        if alt is None:
            experiment_id = client.create_experiment(
                name=alt_name,
                artifact_location=current_store,
            )
            print(f"[DEBUG] Creado experimento '{alt_name}' (ID: {experiment_id})")
        else:
            experiment_id = alt.experiment_id
            print(f"[DEBUG] Usando experimento existente '{alt_name}' (ID: {experiment_id})")
        mlflow.set_experiment(alt_name)
        return experiment_id

    # Si coincide, reutilizamos el experimento
    print(f"[DEBUG] Usando experimento existente '{experiment_name}' (ID: {exp.experiment_id})")
    print(f"[DEBUG] artifact_location: {exp.artifact_location}")
    mlflow.set_experiment(experiment_name)
    return exp.experiment_id


# ------------------------------------------------------
# ENTRENAMIENTO Y REGISTRO DE MODELO
# ------------------------------------------------------

def train_and_log(experiment_id: str, test_size: float = 0.2, seed: int = 42):
    """
    Entrena un modelo simple (LinearRegression) y registra todo en MLflow.
    """
    # Cargar datos
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Entrenar modelo
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = float(mean_squared_error(y_test, preds))

    print(f"[INFO] MSE (test): {mse:.4f}")

    # Registrar en MLflow
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        print(f"[DEBUG] run_id: {run_id}")
        print(f"[DEBUG] artifact_uri: {run.info.artifact_uri}")

        # Log de parámetros y métricas
        mlflow.log_params({
            "model": "LinearRegression",
            "test_size": test_size,
            "random_state": seed,
        })
        mlflow.log_metric("mse", mse)

        # Log de artefactos adicionales (archivo JSON)
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        metrics_file = artifacts_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump({"mse": mse}, f, indent=2)
        mlflow.log_artifact(str(metrics_file))

        # Registrar el modelo (Sklearn flavor)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="models",
        )

    print("✅ Entrenamiento y registro completados.")


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------

if __name__ == "__main__":
    # Configurar MLflow local
    ensure_tracking_and_registry()

    # Leer nombre de experimento desde variable o usar por defecto
    EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "CI-CD-Lab2")
    experiment_id = get_or_create_experiment(EXPERIMENT_NAME)

    # Entrenar y registrar modelo
    train_and_log(experiment_id=experiment_id)
