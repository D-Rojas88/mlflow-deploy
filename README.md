Taller CI/CD con MLflow y GitHub Actions

Presentado por: Daniela Rojas Silva - Geraldine Patiño

Proyecto: Taller CI/CD – Entrenamiento y Validación de un Modelo con MLflow

Descripción del Proyecto

Este proyecto implementa un pipeline de entrenamiento, registro y validación de un modelo de Machine Learning utilizando MLflow y GitHub Actions como parte del taller de integración continua y despliegue continuo (CI/CD).

El objetivo es automatizar el proceso completo de entrenamiento y validación del modelo, garantizando trazabilidad, reproducibilidad y registro de métricas.

Estructura del Proyecto

mlflow-deploy/

├── train.py                → Entrena y registra el modelo en MLflow

├── validate.py             → Carga y valida el modelo entrenado

├── requirements.txt        → Dependencias del entorno

├── Makefile                → Comandos automatizados para entrenamiento y validación

├── .github/

│   └── workflows/

│       └── mlflow-ci.yml   → Pipeline de GitHub Actions para CI/CD

└── mlruns/                 → Carpeta local de tracking de MLflow

Ejecución Local

Crear entorno virtual:

python -m venv .venv
source .venv/bin/activate       # En Windows: .venv\Scripts\activate


Instalar dependencias:

pip install -r requirements.txt


Entrenar y registrar modelo:

make train


Validar modelo entrenado:

make validate


Durante la ejecución, MLflow registra automáticamente las métricas, parámetros, artefactos (model.pkl, metrics.json) y el experimento local en la carpeta mlruns/.

Pipeline CI/CD con GitHub Actions

El workflow definido en .github/workflows/mlflow-ci.yml automatiza el proceso cada vez que se hace push a la rama principal (main).

Etapas del Pipeline

Instalación del entorno

Configura Python 3.11

Instala dependencias del archivo requirements.txt

Entrenamiento (train.py)

Ejecuta make train

Entrena un modelo LinearRegression con el dataset de diabetes

Registra parámetros, métricas y artefactos en MLflow

Validación (validate.py)

Ejecuta make validate

Carga el modelo y evalúa el error cuadrático medio (MSE)

Publicación de artefactos

Sube el archivo model.pkl validado

Sube la carpeta mlruns/ como auditoría de métricas

Resultados del Modelo
Métrica	Valor aproximado
MSE (test)	2900.19
Modelo	Linear Regression
Dataset	Diabetes Dataset (de sklearn)
Experimento MLflow	CI-CD-Lab2-ci

Los resultados y artefactos pueden visualizarse desde la pestaña Actions → Artifacts en GitHub.

Makefile

El proyecto cuenta con un Makefile simple que automatiza la ejecución de los scripts principales:

.PHONY: help train validate

help:
	@echo "make train     - Entrena y registra el modelo"
	@echo "make validate  - Valida el modelo"

train:
	python train.py

validate:
	python validate.py

Criterios Cumplidos

Organización del proyecto: estructura clara y coherente.

Entrenamiento y registro del modelo: incluye firma, ejemplo de entrada y métrica MSE.

Validación del modelo: carga y evaluación correctas del modelo registrado.

Uso del Makefile: comandos make train y make validate funcionando correctamente.

Registro en MLflow: tracking funcional con artefactos auditables.

Pipeline CI/CD: ejecución automática con GitHub Actions.

Código limpio y documentado.

Conclusión

El proyecto cumple con los requerimientos del taller de CI/CD, demostrando un flujo completo de integración y despliegue continuo para un modelo de Machine Learning utilizando MLflow y GitHub Actions.
