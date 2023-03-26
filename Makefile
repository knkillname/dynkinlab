PIPENV_VENV_IN_PROJECT := 1
export PIPENV_VENV_IN_PROJECT

.venv: Pipfile
	pipenv install --dev
