PIPENV_VENV_IN_PROJECT := 1
export PIPENV_VENV_IN_PROJECT

.venv: Pipfile
	pipenv install --dev

.PHONY: format
format: .venv
	pipenv run isort .
	pipenv run black .
