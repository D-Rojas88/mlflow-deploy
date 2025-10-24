.PHONY: help train validate

help:
	@echo "make train     - Entrena y registra el modelo"
	@echo "make validate  - Valida el modelo"

train:
	python train.py

validate:
	python validate.py
